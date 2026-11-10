import {lv95ToLngLat, lngLatToLV95, lv95ToWorld, mercatorFromLngLat} from './lv95.mjs';

// swisstopo's WMTS tile grid in EPSG:2056, tile matrix set 2056_27 from
// https://wmts.geo.admin.ch/EPSG/2056/1.0.0/WMTSCapabilities.xml. Its resolutions are not powers of two, so MapLibre's
// tile sources, which need a quad tree, cannot load it; this custom layer does.
const ORIGIN = [2420000, 1350000];
const EXTENT = [480000, 320000];
const TILE_SIZE = 256;
const RESOLUTIONS = [
    4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250, 2000, 1750,
    1500, 1250, 1000, 750, 650, 500, 250, 100, 50, 20,
    10, 5, 2.5, 2, 1.5, 1, 0.5, 0.25
];
const TILE_URL = 'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/2056/{z}/{x}/{y}.jpeg';

/** The most tiles drawn in one frame; pitched views fall back to coarser levels to stay under it. */
const MAX_TILES = 300;
/** The most textures kept, least recently used ones are dropped first. */
const MAX_TEXTURES = 400;

const VERTEX_SHADER = `#version 300 es
precision highp float;
in vec2 a_pos;
in vec2 a_uv;
uniform mat4 u_matrix;
uniform vec2 u_tile_origin;
uniform vec4 u_uv_transform;
out vec2 v_uv;
void main() {
    v_uv = u_uv_transform.xy + a_uv * u_uv_transform.zw;
    gl_Position = u_matrix * vec4(a_pos + u_tile_origin, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision mediump float;
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 fragColor;
void main() {
    fragColor = texture(u_texture, v_uv);
}`;

/** [E, N] of the top-left corner of a tile and its side in meters. */
function tileBounds({z, x, y}) {
    const size = RESOLUTIONS[z] * TILE_SIZE;
    return {west: ORIGIN[0] + x * size, north: ORIGIN[1] - y * size, size};
}

/** The number of tile columns and rows of a level, as the capabilities document lists them. */
function matrixSize(z) {
    const size = RESOLUTIONS[z] * TILE_SIZE;
    return [Math.ceil(EXTENT[0] / size), Math.ceil(EXTENT[1] / size)];
}

/**
 * Draws swisstopo's national map in color, cut in the LV95 grid, in both the `lv95` and the `mercator` projection.
 *
 * Each tile is a mesh whose vertices are LV95 grid points placed in the world plane of the current projection.
 * In LV95 that plane is the LV95 grid itself, scaled, so a tile is an exact square and needs a single quad: the
 * imagery is shown without any resampling beyond the camera's own. In Web Mercator the grid is curved and turned,
 * so a tile becomes a 16 x 16 mesh of reprojected points, as any LV95 imagery on a Web Mercator map must.
 *
 * Positions are sent relative to the tile's corner and the map center, and the matrix takes the map center in double
 * precision, so that tiles stay in place at street level, where world coordinates exceed single precision.
 *
 * The layer loads the tiles of the view whether it is shown or not, so that showing it needs no loading. Shown, it
 * clears what the layers below it drew, so it is meant to be the top layer of the style.
 */
export class SwisstopoWmtsLayer {
    constructor(id) {
        this.id = id;
        this.type = 'custom';
        this.renderingMode = '2d';
        this.shown = true;
        this.meshes = new Map();
        this.textures = new Map();
        this.pending = new Map();
        this.stats = {level: 0, resolution: RESOLUTIONS[0], tiles: 0};
    }

    setShown(shown) {
        this.shown = shown;
        this.map?.triggerRepaint();
    }

    onAdd(map, gl) {
        this.map = map;
        this.gl = gl;
        this.anisotropic = gl.getExtension('EXT_texture_filter_anisotropic');
        const compile = (type, source) => {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
            return shader;
        };
        this.program = gl.createProgram();
        gl.attachShader(this.program, compile(gl.VERTEX_SHADER, VERTEX_SHADER));
        gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
        gl.linkProgram(this.program);
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
        this.locations = Object.fromEntries(['a_pos', 'a_uv'].map(name => [name, gl.getAttribLocation(this.program, name)]));
        for (const name of ['u_matrix', 'u_tile_origin', 'u_uv_transform', 'u_texture']) {
            this.locations[name] = gl.getUniformLocation(this.program, name);
        }
    }

    onRemove() {
        const gl = this.gl;
        for (const mesh of this.meshes.values()) {
            gl.deleteVertexArray(mesh.vao);
            for (const buffer of mesh.buffers) gl.deleteBuffer(buffer);
        }
        this.meshes.clear();
        for (const controller of this.pending.values()) controller.abort();
        this.pending.clear();
        for (const texture of this.textures.values()) gl.deleteTexture(texture.handle);
        this.textures.clear();
        gl.deleteProgram(this.program);
    }

    /** Where an LV95 grid point lies in the world square of the current projection. */
    _toWorld(projection, east, north) {
        return projection === 'lv95' ? lv95ToWorld(east, north) : mercatorFromLngLat(...lv95ToLngLat(east, north));
    }

    /** The mesh of a tile in the current projection, built when first drawn. */
    _getMesh(projection, tile) {
        const key = `${projection}/${tile.z}/${tile.x}/${tile.y}`;
        let mesh = this.meshes.get(key);
        if (mesh) return mesh;
        const gl = this.gl;
        const grid = projection === 'lv95' ? 1 : 16;
        const {west, north, size} = tileBounds(tile);
        const origin = this._toWorld(projection, west, north);
        const positions = new Float32Array((grid + 1) ** 2 * 2);
        const uvs = new Float32Array((grid + 1) ** 2 * 2);
        for (let j = 0, i = 0; j <= grid; j++) {
            for (let k = 0; k <= grid; k++, i++) {
                const [worldX, worldY] = this._toWorld(projection, west + size * k / grid, north - size * j / grid);
                positions.set([worldX - origin[0], worldY - origin[1]], 2 * i);
                uvs.set([k / grid, j / grid], 2 * i);
            }
        }
        const triangles = [];
        for (let j = 0; j < grid; j++) {
            for (let k = 0; k < grid; k++) {
                const a = j * (grid + 1) + k;
                triangles.push(a, a + 1, a + grid + 1, a + 1, a + grid + 2, a + grid + 1);
            }
        }

        mesh = {origin, vao: gl.createVertexArray(), buffers: [], triangleCount: triangles.length};
        gl.bindVertexArray(mesh.vao);
        for (const [location, data] of [[this.locations.a_pos, positions], [this.locations.a_uv, uvs]]) {
            const buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
            gl.enableVertexAttribArray(location);
            gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
            mesh.buffers.push(buffer);
        }
        const indices = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indices);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(triangles), gl.STATIC_DRAW);
        mesh.buffers.push(indices);
        gl.bindVertexArray(null);
        this.meshes.set(key, mesh);
        return mesh;
    }

    /** The meters one CSS pixel spans at the center of the screen, measured along both screen axes. */
    _metersPerPixel() {
        const canvas = this.map.getCanvas();
        const x = canvas.clientWidth / 2;
        const y = canvas.clientHeight / 2;
        const toLV95 = (point) => {
            const lngLat = this.map.unproject(point);
            return lngLatToLV95(lngLat.lng, lngLat.lat);
        };
        const center = toLV95([x, y]);
        const right = toLV95([x + 1, y]);
        const down = toLV95([x, y + 1]);
        return (Math.hypot(right[0] - center[0], right[1] - center[1]) + Math.hypot(down[0] - center[0], down[1] - center[1])) / 2;
    }

    /**
     * The level whose pixels come closest to the screen's CSS pixels, so that the map's text and symbols keep the size
     * they were designed at, as MapLibre shows the 256 px tiles of a raster source.
     */
    _pickLevel() {
        const metersPerPixel = this._metersPerPixel();
        let best = 0;
        for (let z = 1; z < RESOLUTIONS.length; z++) {
            if (Math.abs(Math.log(RESOLUTIONS[z] / metersPerPixel)) < Math.abs(Math.log(RESOLUTIONS[best] / metersPerPixel))) best = z;
        }
        return best;
    }

    /** The tiles of a level under the screen, nearest the center first: the LV95 box around a grid of screen points. */
    _tilesInView(z) {
        const canvas = this.map.getCanvas();
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        let minE = Infinity, minN = Infinity, maxE = -Infinity, maxN = -Infinity;
        for (let j = 0; j <= 6; j++) {
            for (let i = 0; i <= 6; i++) {
                const lngLat = this.map.unproject([width * i / 6, height * j / 6]);
                const [east, north] = lngLatToLV95(lngLat.lng, lngLat.lat);
                minE = Math.min(minE, east);
                maxE = Math.max(maxE, east);
                minN = Math.min(minN, north);
                maxN = Math.max(maxN, north);
            }
        }
        const center = this.map.getCenter();
        const [centerE, centerN] = lngLatToLV95(center.lng, center.lat);
        for (; z >= 0; z--) {
            const size = RESOLUTIONS[z] * TILE_SIZE;
            const [columns, rows] = matrixSize(z);
            const x0 = Math.max(0, Math.floor((minE - ORIGIN[0]) / size));
            const x1 = Math.min(columns - 1, Math.floor((maxE - ORIGIN[0]) / size));
            const y0 = Math.max(0, Math.floor((ORIGIN[1] - maxN) / size));
            const y1 = Math.min(rows - 1, Math.floor((ORIGIN[1] - minN) / size));
            if (x1 < x0 || y1 < y0) return [];
            if ((x1 - x0 + 1) * (y1 - y0 + 1) > MAX_TILES && z > 0) continue;
            const tiles = [];
            for (let y = y0; y <= y1; y++) {
                for (let x = x0; x <= x1; x++) {
                    const dx = x + 0.5 - (centerE - ORIGIN[0]) / size;
                    const dy = y + 0.5 - (ORIGIN[1] - centerN) / size;
                    tiles.push({z, x, y, distance: dx * dx + dy * dy});
                }
            }
            return tiles.sort((a, b) => a.distance - b.distance);
        }
        return [];
    }

    _loadTexture(tile) {
        const key = `${tile.z}/${tile.x}/${tile.y}`;
        const texture = this.textures.get(key);
        if (texture) {
            texture.lastUsed = performance.now();
            return;
        }
        if (this.pending.has(key)) return;
        const controller = new AbortController();
        this.pending.set(key, controller);
        fetch(TILE_URL.replace('{z}', tile.z).replace('{x}', tile.x).replace('{y}', tile.y), {signal: controller.signal})
            .then(response => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                return response.blob();
            })
            .then(blob => createImageBitmap(blob))
            .then(bitmap => {
                this.pending.delete(key);
                const gl = this.gl;
                const handle = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, handle);
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
                gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
                gl.generateMipmap(gl.TEXTURE_2D);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                if (this.anisotropic) gl.texParameterf(gl.TEXTURE_2D, this.anisotropic.TEXTURE_MAX_ANISOTROPY_EXT, 16);
                this.textures.set(key, {handle, tile, lastUsed: performance.now()});
                this._evictTextures();
                this.map.triggerRepaint();
            })
            .catch(error => {
                this.pending.delete(key);
                if (error.name !== 'AbortError') console.warn(`swisstopo tile ${key} failed to load:`, error.message);
            });
    }

    _evictTextures() {
        if (this.textures.size <= MAX_TEXTURES) return;
        const oldest = [...this.textures.entries()].sort((a, b) => a[1].lastUsed - b[1].lastUsed);
        while (this.textures.size > MAX_TEXTURES) {
            const [key, texture] = oldest.shift();
            this.gl.deleteTexture(texture.handle);
            this.textures.delete(key);
        }
    }

    /**
     * The texture to draw a tile with: its own, or else the loaded tile of the nearest coarser level that contains it,
     * with the part of its image the tile covers, as `[u, v, width, height]`.
     */
    _findTexture(tile) {
        const own = this.textures.get(`${tile.z}/${tile.x}/${tile.y}`);
        if (own) return {texture: own, uv: [0, 0, 1, 1]};
        const target = tileBounds(tile);
        let best = null;
        for (const texture of this.textures.values()) {
            if (texture.tile.z >= tile.z || (best && texture.tile.z <= best.texture.tile.z)) continue;
            const source = tileBounds(texture.tile);
            const u = (target.west - source.west) / source.size;
            const v = (source.north - target.north) / source.size;
            const scale = target.size / source.size;
            if (u < -1e-9 || v < -1e-9 || u + scale > 1 + 1e-9 || v + scale > 1 + 1e-9) continue;
            best = {texture, uv: [u, v, scale, scale]};
        }
        return best;
    }

    render(gl, args) {
        const level = this._pickLevel();
        const tiles = this._tilesInView(level);
        this.stats = {level: tiles[0]?.z ?? level, resolution: RESOLUTIONS[tiles[0]?.z ?? level], tiles: tiles.length};
        for (const tile of tiles) this._loadTexture(tile);
        if (!this.shown) return;

        gl.clearColor(1, 1, 1, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

        const projection = this.map.getProjection()?.type === 'lv95' ? 'lv95' : 'mercator';
        const worldSize = 512 * Math.pow(2, this.map.getZoom());
        const center = this.map.getCenter();
        const scene = projection === 'lv95' ? lv95ToWorld(...lngLatToLV95(center.lng, center.lat)) : mercatorFromLngLat(center.lng, center.lat);
        const matrix = new Float64Array(16);
        const viewProjection = args.modelViewProjectionMatrix;
        for (let row = 0; row < 4; row++) {
            matrix[row] = viewProjection[row] * worldSize;
            matrix[4 + row] = viewProjection[4 + row] * worldSize;
            matrix[8 + row] = viewProjection[8 + row];
            matrix[12 + row] = viewProjection[row] * scene[0] * worldSize + viewProjection[4 + row] * scene[1] * worldSize + viewProjection[12 + row];
        }

        gl.useProgram(this.program);
        gl.uniformMatrix4fv(this.locations.u_matrix, false, new Float32Array(matrix));
        gl.uniform1i(this.locations.u_texture, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

        const drawn = tiles
            .map(tile => ({tile, found: this._findTexture(tile)}))
            .filter(({found}) => found)
            .sort((a, b) => a.found.texture.tile.z - b.found.texture.tile.z);
        for (const {tile, found} of drawn) {
            const mesh = this._getMesh(projection, tile);
            found.texture.lastUsed = performance.now();
            gl.bindVertexArray(mesh.vao);
            gl.bindTexture(gl.TEXTURE_2D, found.texture.handle);
            gl.uniform2f(this.locations.u_tile_origin, mesh.origin[0] - scene[0], mesh.origin[1] - scene[1]);
            gl.uniform4f(this.locations.u_uv_transform, ...found.uv);
            gl.drawElements(gl.TRIANGLES, mesh.triangleCount, gl.UNSIGNED_SHORT, 0);
        }
        gl.bindVertexArray(null);
    }
}
