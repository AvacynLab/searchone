export const logClientEvent = (type, payload = {}) => {
  try {
    console.info(`[searchone:${type}]`, payload)
  } catch (e) {
    /* noop */
  }
}
