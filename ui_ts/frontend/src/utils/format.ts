export const currency = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
};

export const pct = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
};

export const shares = (value: number): string => value.toFixed(4).replace(/\.?0+$/, "");
