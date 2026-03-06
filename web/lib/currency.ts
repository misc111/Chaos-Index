type CurrencyOptions = {
  minimumFractionDigits?: number;
  maximumFractionDigits?: number;
};

export function formatUsd(value: number, options: CurrencyOptions = {}): string {
  const minimumFractionDigits = options.minimumFractionDigits ?? 0;
  const maximumFractionDigits = options.maximumFractionDigits ?? minimumFractionDigits;

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits,
    maximumFractionDigits,
  }).format(value);
}

export function formatSignedUsd(value: number, options: CurrencyOptions = {}): string {
  const prefix = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${prefix}${formatUsd(Math.abs(value), options)}`;
}
