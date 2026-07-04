import type { DemographicFilter as DemographicFilterValue, HomeWorkResponse } from "../api";

const EMPTY: DemographicFilterValue = { gender: null, age_bracket: null, job: null };

function label(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function DemographicFilter({
  options,
  value,
  onChange,
}: {
  options: HomeWorkResponse["filter_options"];
  value: DemographicFilterValue;
  onChange: (next: DemographicFilterValue) => void;
}) {
  const isEmpty = !value.gender && !value.age_bracket && !value.job;

  return (
    <div className="hw-filter" aria-label="Demographic filter">
      <select
        aria-label="Gender"
        value={value.gender ?? ""}
        onChange={(e) => onChange({ ...value, gender: e.target.value || null })}
      >
        <option value="">Any gender</option>
        {options.genders.map((g) => (
          <option key={g} value={g}>
            {label(g)}
          </option>
        ))}
      </select>
      <select
        aria-label="Age bracket"
        value={value.age_bracket ?? ""}
        onChange={(e) => onChange({ ...value, age_bracket: e.target.value || null })}
      >
        <option value="">Any age</option>
        {options.age_brackets.map((b) => (
          <option key={b.key} value={b.key}>
            {b.label}
          </option>
        ))}
      </select>
      <select
        aria-label="Job"
        value={value.job ?? ""}
        onChange={(e) => onChange({ ...value, job: e.target.value || null })}
      >
        <option value="">Any job</option>
        {options.jobs.map((j) => (
          <option key={j} value={j}>
            {label(j)}
          </option>
        ))}
      </select>
      {!isEmpty && (
        <button className="btn btn-secondary" style={{ padding: "6px 12px", fontSize: 13 }} onClick={() => onChange(EMPTY)}>
          Reset
        </button>
      )}
    </div>
  );
}
