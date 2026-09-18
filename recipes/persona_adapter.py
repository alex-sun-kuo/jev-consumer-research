"""Map user-selected fields from CSV, JSON, JSONL, or Python records into a study."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path

from recipes.run_study import validate_study


def read_personas(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("CSV headers must be present and unique")
            records = list(reader)
            if any(None in row for row in records):
                raise ValueError("CSV row has more values than headers")
        elif path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in stream if line.strip()]
        elif path.suffix.lower() == ".json":
            records = json.load(stream)
        else:
            raise ValueError("Use .csv, .json (an array), or .jsonl")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Persona data must be a list of objects")
    return records


def source_value(record, field):
    # Exact column names take precedence over dotted paths into nested JSON.
    if field in record:
        return record[field]
    value = record
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def convert_value(value, kind):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if kind is None:
        # Preserve JSON arrays/objects and existing numbers without guessing CSV types.
        json.dumps(value, allow_nan=False)
        return copy.deepcopy(value)
    if kind == "str":
        return str(value)
    if kind == "bool":
        if type(value) is bool:
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError("Expected true or false")
    if kind in ("float", "int"):
        if isinstance(value, bool):
            raise ValueError("Boolean is not a number")
        number = float(value)
        if not math.isfinite(number) or (kind == "int" and not number.is_integer()):
            raise ValueError("Expected a finite number of the requested type")
        return int(number) if kind == "int" else number
    raise ValueError(f"Unknown field type: {kind}")


def mapped_fields(mapping):
    """Validate the mapping shape; return {'section.target': source} and its types."""
    if not isinstance(mapping, dict) or not isinstance(mapping.get("id_field"), str) or not mapping["id_field"]:
        raise ValueError("Mapping requires id_field")
    fields = {}
    for section in ("demographics", "context"):
        selected = mapping.get(section, {})
        if not isinstance(selected, dict):
            raise ValueError(f"{section} must map field names to source columns or paths")
        for target, origin in selected.items():
            if not isinstance(target, str) or not target or "." in target or not isinstance(origin, str) or not origin:
                raise ValueError("Use nonempty field names (without dots) and source paths")
            fields[f"{section}.{target}"] = origin
    types = mapping.get("types", {})
    if not isinstance(types, dict) or set(types) - set(fields):
        raise ValueError("Type declarations must name mapped fields, e.g. context.budget_usd")
    if any(kind not in ("str", "float", "int", "bool") for kind in types.values()):
        raise ValueError("Types must be str, float, int, or bool")
    return fields, types


def field_exists(record, origin):
    # A column explicitly present but entirely null is still a valid unknown field.
    if origin in record:
        return True
    for part in origin.split("."):
        if not isinstance(record, dict) or part not in record:
            return False
        record = record[part]
    return True


def adapt_personas(records, mapping, *, source="Python records"):
    """Mapping dictionaries use destination field names as keys and source paths as values."""
    if not isinstance(records, list) or not records or not all(isinstance(row, dict) for row in records):
        raise ValueError("Provide a nonempty list of persona objects")
    fields, types = mapped_fields(mapping)
    for origin in (mapping["id_field"], *fields.values()):
        if not any(field_exists(row, origin) for row in records):
            raise ValueError(f"Source field not found: {origin}")
    profiles, seen = [], set()
    for index, record in enumerate(records, 1):
        identifier = source_value(record, mapping["id_field"])
        if type(identifier) not in (str, int) or not str(identifier).strip():
            raise ValueError(f"Row {index}: missing or invalid persona ID")
        identifier = str(identifier).strip()
        if identifier in seen:
            raise ValueError(f"Row {index}: duplicate persona ID")
        seen.add(identifier)
        profile = {"id": identifier, "demographics": {}, "context": {}}
        for target, origin in fields.items():
            section, key = target.split(".")
            try:
                profile[section][key] = convert_value(source_value(record, origin), types.get(target))
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"Row {index}, {target}: invalid value for mapped type") from exc
        profile["provenance"] = {"kind": "user_supplied_unverified", "source": str(source),
                                 "field_mapping": copy.deepcopy(mapping),
                                 "note": "Mapped values only; no demographic or preference inference."}
        profiles.append(profile)
    return profiles


def with_personas(study, profiles, *, revision):
    """Keep the stimulus design, replace the panel, and validate before running."""
    if not isinstance(revision, str) or not revision.strip() or revision == study.get("revision"):
        raise ValueError("Use a new nonempty study revision for the imported panel")
    result = copy.deepcopy(study)
    result.update(profiles=copy.deepcopy(profiles), revision=revision)
    result["provenance"] = "User-supplied panel; see profile provenance. Stimuli retain the source study's design."
    validate_study(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("personas", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "study.json")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = adapt_personas(read_personas(args.personas), json.loads(args.mapping.read_text()), source=args.personas.name)
    study = with_personas(json.loads(args.study.read_text()), profiles, revision=args.revision)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(study, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Imported {len(profiles)} profiles into {args.output}")


if __name__ == "__main__":
    main()
