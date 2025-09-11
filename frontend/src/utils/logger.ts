const isProd = process.env.NODE_ENV === 'production';

export const logger = {
  debug: (..._args: unknown[]) => {
    if (!isProd) console.debug(..._args); // eslint-disable-line no-console
  },
  info: (..._args: unknown[]) => {
    if (!isProd) console.info(..._args); // eslint-disable-line no-console
  },
  warn: (...args: unknown[]) => console.warn(...args), // eslint-disable-line no-console
  error: (...args: unknown[]) => console.error(...args), // eslint-disable-line no-console
};
