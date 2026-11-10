// WGS84 <-> LV95 (EPSG:2056) with swisstopo's rigorous formulas, and the LV95 world plane MapLibre's `lv95`
// projection moves its camera in. This mirrors src/geo/projection/lv95_utils.ts formula for formula, so that the
// example places swisstopo's LV95 tiles exactly where the projection draws the rest of the map.

const WGS84_A = 6378137;
const WGS84_E2 = 0.00669437999014;
const BESSEL_A = 6377397.155;
const BESSEL_E2 = 0.006674372230614;
const BESSEL_E = Math.sqrt(BESSEL_E2);
const CH1903PLUS_TO_WGS84 = [674.374, 15.056, 405.346];

const PHI0 = (46 + 57 / 60 + 8.66 / 3600) * Math.PI / 180;
const LAMBDA0 = (7 + 26 / 60 + 22.5 / 3600) * Math.PI / 180;
const R = BESSEL_A * Math.sqrt(1 - BESSEL_E2) / (1 - BESSEL_E2 * Math.sin(PHI0) ** 2);
const ALPHA = Math.sqrt(1 + BESSEL_E2 / (1 - BESSEL_E2) * Math.cos(PHI0) ** 4);
const B0 = Math.asin(Math.sin(PHI0) / ALPHA);
const K = Math.atanh(Math.sin(B0)) - ALPHA * Math.atanh(Math.sin(PHI0)) + ALPHA * BESSEL_E * Math.atanh(BESSEL_E * Math.sin(PHI0));
const SIN_B0 = Math.sin(B0);
const COS_B0 = Math.cos(B0);

const E0 = 2600000;
const N0 = 1200000;

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

function geocentric(lambda, phi, height, a, e2) {
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const n = a / Math.sqrt(1 - e2 * sinPhi * sinPhi);
    return [(n + height) * cosPhi * Math.cos(lambda), (n + height) * cosPhi * Math.sin(lambda), (n * (1 - e2) + height) * sinPhi];
}

function geodetic(x, y, z, a, e2) {
    const p = Math.hypot(x, y);
    let phi = Math.atan2(z, p * (1 - e2));
    let height = 0;
    for (let i = 0; i < 5; i++) {
        const sinPhi = Math.sin(phi);
        const n = a / Math.sqrt(1 - e2 * sinPhi * sinPhi);
        height = Math.abs(Math.cos(phi)) > 1e-9 ? p / Math.cos(phi) - n : Math.abs(z) - n * (1 - e2);
        phi = Math.atan2(z, p * (1 - e2 * n / (n + height)));
    }
    return [Math.atan2(y, x), phi, height];
}

function wgs84ToCH1903Plus(lambda, phi) {
    const [x, y, z] = geocentric(lambda, phi, 0, WGS84_A, WGS84_E2);
    const [lambdaB, phiB] = geodetic(x - CH1903PLUS_TO_WGS84[0], y - CH1903PLUS_TO_WGS84[1], z - CH1903PLUS_TO_WGS84[2], BESSEL_A, BESSEL_E2);
    return [lambdaB, phiB];
}

function ch1903PlusToWgs84(lambda, phi) {
    let heightB = 0;
    let result;
    for (let i = 0; i < 3; i++) {
        const [x, y, z] = geocentric(lambda, phi, heightB, BESSEL_A, BESSEL_E2);
        result = geodetic(x + CH1903PLUS_TO_WGS84[0], y + CH1903PLUS_TO_WGS84[1], z + CH1903PLUS_TO_WGS84[2], WGS84_A, WGS84_E2);
        heightB -= result[2];
    }
    return [result[0], result[1]];
}

function ch1903PlusToLV95Offset(lambda, phi) {
    const s = ALPHA * Math.atanh(Math.sin(phi)) - ALPHA * BESSEL_E * Math.atanh(BESSEL_E * Math.sin(phi)) + K;
    const sinB = Math.tanh(s);
    const cosB = 1 / Math.cosh(s);
    const l = ALPHA * (lambda - LAMBDA0);
    const sinObliqueB = clamp(COS_B0 * sinB - SIN_B0 * cosB * Math.cos(l), -1, 1);
    const obliqueL = Math.atan2(cosB * Math.sin(l), SIN_B0 * sinB + COS_B0 * cosB * Math.cos(l));
    return [R * obliqueL, R * Math.atanh(sinObliqueB)];
}

function lv95OffsetToCH1903Plus(east, north) {
    const obliqueL = east / R;
    const sinObliqueB = Math.tanh(north / R);
    const cosObliqueB = 1 / Math.cosh(north / R);
    const sinB = clamp(COS_B0 * sinObliqueB + SIN_B0 * cosObliqueB * Math.cos(obliqueL), -1, 1);
    const l = Math.atan2(cosObliqueB * Math.sin(obliqueL), COS_B0 * cosObliqueB * Math.cos(obliqueL) - SIN_B0 * sinObliqueB);
    const lambda = LAMBDA0 + l / ALPHA;
    if (Math.abs(sinB) >= 1) return [lambda, Math.sign(sinB) * Math.PI / 2];
    const psi = (Math.atanh(sinB) - K) / ALPHA;
    let phi = Math.asin(sinB);
    for (let i = 0; i < 8; i++) {
        const next = 2 * Math.atan(Math.exp(psi + BESSEL_E * Math.atanh(BESSEL_E * Math.sin(phi)))) - Math.PI / 2;
        const done = Math.abs(next - phi) < 1e-14;
        phi = next;
        if (done) break;
    }
    return [lambda, phi];
}

/** WGS84 [lng, lat] in degrees to LV95 [E, N] in meters. */
export function lngLatToLV95(lng, lat) {
    const [lambda, phi] = wgs84ToCH1903Plus(lng * Math.PI / 180, lat * Math.PI / 180);
    const [east, north] = ch1903PlusToLV95Offset(lambda, phi);
    return [E0 + east, N0 + north];
}

/** LV95 [E, N] in meters to WGS84 [lng, lat] in degrees. */
export function lv95ToLngLat(east, north) {
    const [lambda, phi] = lv95OffsetToCH1903Plus(east - E0, north - N0);
    const [lng, lat] = ch1903PlusToWgs84(lambda, phi);
    return [lng * 180 / Math.PI, lat * 180 / Math.PI];
}

/** Web Mercator x and y in the 0..1 world square, with the same expressions as MapLibre's MercatorCoordinate. */
export function mercatorFromLngLat(lng, lat) {
    return [(180 + lng) / 360, (180 - (180 / Math.PI * Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)))) / 360];
}

const ORIGIN_LNG_LAT = lv95ToLngLat(E0, N0);
const [ORIGIN_X, ORIGIN_Y] = mercatorFromLngLat(...ORIGIN_LNG_LAT);
const UNITS_PER_METER = (mercatorFromLngLat(...lv95ToLngLat(E0 + 1, N0))[0] - mercatorFromLngLat(...lv95ToLngLat(E0 - 1, N0))[0]) / 2;

/**
 * LV95 [E, N] in meters to the world plane of MapLibre's `lv95` projection: easting along x, northing against y,
 * scaled to match Web Mercator at Bern, with Bern at its Web Mercator position.
 */
export function lv95ToWorld(east, north) {
    return [ORIGIN_X + (east - E0) * UNITS_PER_METER, ORIGIN_Y - (north - N0) * UNITS_PER_METER];
}
