// Swiss LV95 projection: Web Mercator tiles are reprojected into the LV95 world plane per vertex.
// The constants mirror src/geo/projection/lv95_utils.ts, which projection_lv95.test.ts checks.
//
// `u_projection_reprojection_params.x` selects how a tile reaches the plane:
//   0: tile coordinates go through `u_projection_matrix` unchanged, as for terrain render-to-texture.
//   1: the exact projection of the tile's mercator position, for the zoom levels where float precision is ample.
//   2: a biquadratic polynomial the CPU fits around a reference point, which keeps street-level zooms precise.
// Modes 1 and 2 yield planar tile units: world units relative to `u_projection_reprojection_origin.xy`,
// divided by the world size of one tile unit, which `u_projection_matrix` takes to clip space.

const float LV95_ALPHA = 1.000729138;
const float LV95_SIN_B0 = 0.7302544705;
const float LV95_COS_B0 = 0.6831752398;
const float LV95_K = 0.003066732377;
const float LV95_LAMBDA0 = 0.1298452241;
const float LV95_BESSEL_E = 0.08169683122;
const vec2 LV95_WORLD_ORIGIN = vec2(0.5206628678, 0.3519253720);
const float LV95_WORLD_UNITS_PER_RADIAN = 0.2327580373;
const float LV95_COS_PHI_ORIGIN = 0.6826225170;

const float LV95_WGS84_A = 6378137.0;
const float LV95_WGS84_E2 = 0.00669437999014;
const float LV95_WGS84_F = 0.003352810665;
// Molodensky shift from WGS84 to CH1903+: geocentric translation, and the differences of the semi-major axis and flattening.
const vec3 LV95_DATUM_SHIFT = vec3(-674.374, -15.056, -405.346);
const float LV95_DATUM_DA = -739.845;
const float LV95_DATUM_DF = -0.00001003748257;

// Beyond these the oblique cylinder runs off to infinity or wraps around, so vertices there are culled.
const float LV95_MAX_OBLIQUE_LONGITUDE = 3.0;
const float LV95_MAX_SIN_OBLIQUE_LATITUDE = 0.99;

bool lv95IsPoleVertex(vec2 rawPos) {
    return rawPos.y < -32767.5 || rawPos.y > 32766.5;
}

// Projects a Web Mercator position to the LV95 world plane with swisstopo's formulas: a Molodensky datum shift to
// CH1903+, the conformal mapping onto the projection sphere, and the oblique cylinder through Bern.
// Also returns the oblique coordinates, and how many world units one mercator unit spans there.
vec2 lv95ExactWorld(vec2 mercator, out float obliqueLongitude, out float sinObliqueLatitude, out float lineScale) {
    float lambda = (mercator.x - 0.5) * 2.0 * PI;
    float psi = PI * (1.0 - 2.0 * mercator.y);
    float sinPhi = tanh(psi);
    float cosPhi = 1.0 / cosh(psi);
    float sinLambda = sin(lambda);
    float cosLambda = cos(lambda);
    float w2 = 1.0 - LV95_WGS84_E2 * sinPhi * sinPhi;
    float n = LV95_WGS84_A / sqrt(w2);
    float m = n * (1.0 - LV95_WGS84_E2) / w2;
    float dPhi = (-LV95_DATUM_SHIFT.x * sinPhi * cosLambda - LV95_DATUM_SHIFT.y * sinPhi * sinLambda + LV95_DATUM_SHIFT.z * cosPhi
        + LV95_DATUM_DA * n * LV95_WGS84_E2 * sinPhi * cosPhi / LV95_WGS84_A
        + LV95_DATUM_DF * (m / (1.0 - LV95_WGS84_F) + n * (1.0 - LV95_WGS84_F)) * sinPhi * cosPhi) / m;
    float dLambda = (-LV95_DATUM_SHIFT.x * sinLambda + LV95_DATUM_SHIFT.y * cosLambda) / (n * cosPhi);
    // The isometric latitude of the shifted latitude, to first order in the shift.
    float psiB = psi + dPhi / cosPhi;
    float sinPhiB = sinPhi + cosPhi * dPhi;
    float s = LV95_ALPHA * (psiB - LV95_BESSEL_E * atanh(LV95_BESSEL_E * sinPhiB)) + LV95_K;
    float sinB = tanh(s);
    float cosB = 1.0 / cosh(s);
    float l = LV95_ALPHA * (lambda + dLambda - LV95_LAMBDA0);
    float cosL = cos(l);
    sinObliqueLatitude = clamp(LV95_COS_B0 * sinB - LV95_SIN_B0 * cosB * cosL, -0.9999999, 0.9999999);
    obliqueLongitude = atan(cosB * sin(l), LV95_SIN_B0 * sinB + LV95_COS_B0 * cosB * cosL);
    lineScale = cosPhi / (LV95_COS_PHI_ORIGIN * sqrt(1.0 - sinObliqueLatitude * sinObliqueLatitude));
    return LV95_WORLD_ORIGIN + vec2(obliqueLongitude, -atanh(sinObliqueLatitude)) * LV95_WORLD_UNITS_PER_RADIAN;
}

// Evaluates the polynomial of mode 2 at a tile position.
vec2 lv95LocalPlanar(vec2 pos) {
    vec2 uv = (pos - u_projection_reprojection_params.zw) * u_projection_reprojection_params.y;
    float u = uv.x;
    float v = uv.y;
    mat4 c = u_projection_reprojection_coefficients;
    return u * (c[0].xy + u * c[1].xy) + v * (c[0].zw + v * c[2].xy) + u * v * (c[1].zw + u * c[2].zw + v * c[3].xy + u * v * c[3].zw);
}

// Projects a tile position to planar tile units. `culled` is set for vertices the projection cannot place,
// and `lineScale` to how many planar tile units one tile unit spans there.
vec2 lv95ProjectToPlane(vec2 pos, vec2 rawPos, out bool culled, out float lineScale) {
    float mode = u_projection_reprojection_params.x;
    lineScale = u_projection_reprojection_origin.z;
    culled = lv95IsPoleVertex(rawPos);
    if (mode < 0.5) {
        return pos;
    }
    if (mode > 1.5) {
        return lv95LocalPlanar(pos);
    }
    vec2 mercator = u_projection_tile_mercator_coords.xy + u_projection_tile_mercator_coords.zw * pos;
    float obliqueLongitude;
    float sinObliqueLatitude;
    vec2 world = lv95ExactWorld(mercator, obliqueLongitude, sinObliqueLatitude, lineScale);
    culled = culled || abs(obliqueLongitude) > LV95_MAX_OBLIQUE_LONGITUDE || abs(sinObliqueLatitude) > LV95_MAX_SIN_OBLIQUE_LATITUDE;
    return (world - u_projection_reprojection_origin.xy) / u_projection_tile_mercator_coords.z;
}

float projectLineThickness(float tileY) {
    return 1.0;
}

float projectCircleRadius(float tileY) {
    return 1.0;
}

vec2 projectTileToPlane(vec2 pos) {
    bool culled;
    float lineScale;
    return lv95ProjectToPlane(pos, pos, culled, lineScale);
}

vec4 projectPlanarTile(vec2 planar, float elevation) {
    return u_projection_matrix * vec4(planar, elevation, 1.0);
}

// Projects a point in tile-local coordinates (usually 0..EXTENT) to screen. Culled vertices, such as pole vertices,
// kill their triangles by being placed far beyond the near plane.
vec4 projectTile(vec2 pos, vec2 rawPos) {
    bool culled;
    float lineScale;
    vec4 result = projectPlanarTile(lv95ProjectToPlane(pos, rawPos, culled, lineScale), 0.0);
    if (culled) {
        result.z = -10000000.0;
    }
    return result;
}

vec4 projectTile(vec2 pos) {
    return projectTile(pos, pos);
}

vec4 projectTileWithElevation(vec2 pos, float elevation) {
    return projectPlanarTile(projectTileToPlane(pos), elevation);
}

vec4 projectTileFor3D(vec2 pos, float elevation) {
    return projectTileWithElevation(pos, elevation);
}

// Extrudes a line in the plane, so that its width in pixels does not change with the scale of the projection.
vec4 projectLineTile(vec2 pos, vec2 extrusion) {
    bool culled;
    float lineScale;
    vec2 planar = lv95ProjectToPlane(pos, pos, culled, lineScale);
    if (extrusion != vec2(0.0)) {
        bool extrudedCulled;
        float extrudedLineScale;
        planar = lv95ProjectToPlane(pos + extrusion / max(lineScale, 1e-3), pos, extrudedCulled, extrudedLineScale);
    }
    vec4 result = projectPlanarTile(planar, 0.0);
    if (culled) {
        result.z = -10000000.0;
    }
    return result;
}
