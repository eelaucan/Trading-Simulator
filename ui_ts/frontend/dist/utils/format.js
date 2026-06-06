export const currency = (value) => {
    if (value === null || value === undefined || Number.isNaN(value))
        return "N/A";
    return new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
    }).format(value);
};
export const pct = (value) => {
    if (value === null || value === undefined || Number.isNaN(value))
        return "N/A";
    return `${(value * 100).toFixed(2)}%`;
};
export const shares = (value) => value.toFixed(4).replace(/\.?0+$/, "");
