import {describe, expect, test} from 'vitest';
import Point from '@mapbox/point-geometry';
import {LngLat} from '../lng_lat.ts';
import {LngLatBounds} from '../lng_lat_bounds.ts';
import {MercatorCoordinate} from '../mercator_coordinate.ts';
import {OverscaledTileID, UnwrappedTileID, CanonicalTileID} from '../../tile/tile_id.ts';
import {EXTENT} from '../../data/extent.ts';
import {createProjectionFromName} from './projection_factory.ts';
import {LV95Projection} from './lv95_projection.ts';
import {LV95Transform, LV95_POLYNOMIAL_MIN_ZOOM, tileWorldBox} from './lv95_transform.ts';
import {lv95WorldCoordinateHelper} from './lv95_utils.ts';
import {coveringTiles} from './covering_tiles.ts';
import {MercatorCameraHelper} from './mercator_camera_helper.ts';
import {createIdentityMat4f32} from '../../util/util.ts';

/** A transform of an 800 x 600 map looking at Switzerland. */
function createTransform(zoom: number, center: LngLat = new LngLat(8.23, 46.82)): LV95Transform {
    const transform = new LV95Transform();
    transform.resize(800, 600);
    transform.setZoom(zoom);
    transform.setCenter(center);
    return transform;
}

/** The tile of a zoom level that contains a location, with the tile coordinates of the location in it. */
function tileAt(location: LngLat, z: number): {tileID: OverscaledTileID; x: number; y: number} {
    const coordinate = MercatorCoordinate.fromLngLat(location);
    const tilesAtZoom = 1 << z;
    const tileX = Math.floor(coordinate.x * tilesAtZoom);
    const tileY = Math.floor(coordinate.y * tilesAtZoom);
    return {
        tileID: new OverscaledTileID(z, 0, z, tileX, tileY),
        x: (coordinate.x * tilesAtZoom - tileX) * EXTENT,
        y: (coordinate.y * tilesAtZoom - tileY) * EXTENT,
    };
}

/** A clip space point in screen pixels. */
function clipToScreen(transform: LV95Transform, point: Point): Point {
    return new Point((point.x + 1) * transform.width / 2, (1 - point.y) * transform.height / 2);
}

describe('LV95 transform', () => {
    test('is what the lv95 projection name selects', () => {
        const {projection, transform, cameraHelper} = createProjectionFromName('lv95', undefined, {});
        expect(projection).toBeInstanceOf(LV95Projection);
        expect(projection.name).toBe('lv95');
        expect(projection.useSubdivision).toBe(true);
        expect(projection.worldCoordinateHelper.wraps).toBe(true);
        expect(transform).toBeInstanceOf(LV95Transform);
        expect(transform.worldCoordinateHelper).toBe(lv95WorldCoordinateHelper);
        expect(transform.tileWorldCoordinateHelper).toBe(projection.worldCoordinateHelper);
        expect(cameraHelper).toBeInstanceOf(MercatorCameraHelper);
        expect(transform.clone()).toBeInstanceOf(LV95Transform);
        projection.destroy();
    });

    test('keeps the center at the center point and round-trips screen points with pitch, bearing, roll and padding', () => {
        const transform = createTransform(9);
        transform.setPadding({left: 80, top: 30, right: 0, bottom: 0});
        transform.setBearing(25);
        transform.setPitch(50);
        transform.setRoll(10);
        for (const zoom of [5, 9, 12, 18]) {
            transform.setZoom(zoom);
            expect(transform.locationToScreenPoint(transform.center).dist(transform.centerPoint)).toBeLessThan(1e-6);
            const location = transform.screenPointToLocation(new Point(300, 400));
            expect(transform.locationToScreenPoint(location).dist(new Point(300, 400))).toBeLessThan(1e-6);
        }
    });

    test('keeps a location under the point it was set at while zooming', () => {
        const transform = createTransform(11);
        transform.setBearing(-35);
        transform.setPitch(30);
        const point = new Point(500, 380);
        const location = transform.screenPointToLocation(point);
        transform.setZoom(12.5);
        transform.setLocationAtPoint(location, point);
        expect(transform.locationToScreenPoint(location).dist(point)).toBeLessThan(1e-6);
    });

    test('reprojects tile coordinates to where the camera shows their location, in both shader modes and far past a tile\'s zoom', () => {
        const zurich = new LngLat(8.5417, 47.3769);
        for (const [zoom, tileZoom] of [[6, 6], [LV95_POLYNOMIAL_MIN_ZOOM - 0.5, 9], [14, 14], [21, 10], [21, 2]]) {
            const transform = createTransform(zoom, zurich);
            transform.setPitch(40);
            const location = new LngLat(zurich.lng + 0.1 / Math.pow(2, zoom - 6), zurich.lat - 0.05 / Math.pow(2, zoom - 6));
            const {tileID, x, y} = tileAt(location, tileZoom);
            const projected = transform.projectTileCoordinates(x, y, tileID.toUnwrapped());
            expect(clipToScreen(transform, projected.point).dist(transform.locationToScreenPoint(location))).toBeLessThan(0.05);
        }
    });

    test('lays pitch-aligned labels out in the plane at the size of tile units and back', () => {
        const transform = createTransform(13, new LngLat(9.5, 47.1));
        const tile = new UnwrappedTileID(0, new CanonicalTileID(13, 4312, 2881));
        const anchor = transform.projectTileCoordinatesToPlane(4000, 4100, tile);
        const tenPixels = 10 * EXTENT / transform.tileSize;
        const projectedAnchor = transform.projectPlanarTileCoordinates(anchor.x, anchor.y, tile).point;
        const right = transform.projectPlanarTileCoordinates(anchor.x + tenPixels, anchor.y, tile).point;
        expect(clipToScreen(transform, projectedAnchor).dist(clipToScreen(transform, transform.projectTileCoordinates(4000, 4100, tile).point))).toBeLessThan(1e-6);
        expect(clipToScreen(transform, right).dist(clipToScreen(transform, projectedAnchor))).toBeCloseTo(10, 6);
        const restored = transform.tileCoordinatesFromPlane(anchor.x, anchor.y, tile);
        expect(restored.dist(new Point(4000, 4100))).toBeLessThan(0.5);
    });

    test('gives screen positions and terrain hits in the Web Mercator grid tiles are cut in', () => {
        const transform = createTransform(12);
        transform.setPitch(45);
        const point = new Point(200, 450);
        const coordinate = transform.screenPointToMercatorCoordinate(point);
        const expected = MercatorCoordinate.fromLngLat(transform.screenPointToLocation(point));
        expect(coordinate.x).toBeCloseTo(expected.x, 12);
        expect(coordinate.y).toBeCloseTo(expected.y, 12);
        expect(transform.getFastPathSimpleProjectionMatrix(new OverscaledTileID(12, 0, 12, 2141, 1440))).toBeNull();
    });

    test('projects tiles exactly in the shader at low zooms, with polynomials from the polynomial zoom on, and passes render-to-texture through', () => {
        const tileID = new OverscaledTileID(9, 0, 9, 267, 180);
        const low = createTransform(LV95_POLYNOMIAL_MIN_ZOOM - 1).getProjectionData({overscaledTileID: tileID});
        expect(low.reprojection.params[0]).toBe(1);
        const high = createTransform(LV95_POLYNOMIAL_MIN_ZOOM).getProjectionData({overscaledTileID: tileID});
        expect(high.reprojection.params[0]).toBe(2);
        expect(high.reprojection.coefficients).toHaveLength(16);
        expect(high.tileMercatorCoords).toEqual([267 / 512, 180 / 512, 1 / 512 / EXTENT, 1 / 512 / EXTENT]);
        tileID.terrainRttPosMatrix32f = createIdentityMat4f32();
        const rtt = createTransform(12).getProjectionData({overscaledTileID: tileID, applyTerrainMatrix: true});
        expect(rtt.reprojection.params[0]).toBe(0);
        expect(rtt.mainMatrix).toBe(tileID.terrainRttPosMatrix32f);
    });

    test('selects the tiles under the visible locations, in the main world only', () => {
        const transform = createTransform(8);
        transform.setPitch(60);
        transform.setBearing(30);
        const tiles = coveringTiles(transform, {tileSize: 512});
        expect(tiles.length).toBeGreaterThan(0);
        expect(tiles.every(tile => tile.wrap === 0)).toBe(true);
        for (const point of [new Point(400, 300), new Point(20, 590), new Point(780, 590), new Point(400, 200)]) {
            const coordinate = transform.screenPointToMercatorCoordinate(point);
            expect(tiles.some(tile => {
                const tilesAtZoom = 1 << tile.canonical.z;
                return tile.canonical.x === Math.floor(coordinate.x * tilesAtZoom) && tile.canonical.y === Math.floor(coordinate.y * tilesAtZoom);
            })).toBe(true);
        }
    });

    test('leaves tiles outside the world square out of the plane', () => {
        expect(tileWorldBox({z: 6, x: 33, y: 22})).not.toBeNull();
        expect(tileWorldBox({z: 4, x: 0, y: 8})).toBeNull();
        const world = tileWorldBox({z: 0, x: 0, y: 0});
        expect(world.minX).toBeLessThanOrEqual(0);
        expect(world.maxX).toBeGreaterThanOrEqual(1);
    });

    test('bounds the view with the curved edges of the viewport', () => {
        const transform = createTransform(5);
        transform.setBearing(20);
        const bounds = transform.getBounds();
        for (let i = 0; i <= 10; i++) {
            const location = transform.screenPointToLocation(new Point(800 * i / 10, 0));
            expect(bounds.contains(location)).toBe(true);
        }
    });

    test('fits a box of Switzerland into the padded view', () => {
        const transform = createTransform(3);
        const helper = new MercatorCameraHelper();
        const bounds = new LngLatBounds([5.96, 45.82], [10.49, 47.81]);
        const padding = {top: 20, bottom: 20, left: 20, right: 20};
        const camera = helper.cameraForBoxAndBearing({maxZoom: 22, offset: [0, 0]}, padding, transform.padding, bounds, 0, transform);
        transform.setZoom(camera.zoom);
        transform.setCenter(camera.center);
        for (const corner of [bounds.getNorthWest(), bounds.getNorthEast(), bounds.getSouthEast(), bounds.getSouthWest()]) {
            const point = transform.locationToScreenPoint(corner);
            expect(point.x).toBeGreaterThanOrEqual(19.9);
            expect(point.x).toBeLessThanOrEqual(780.1);
            expect(point.y).toBeGreaterThanOrEqual(19.9);
            expect(point.y).toBeLessThanOrEqual(580.1);
        }
    });
});
