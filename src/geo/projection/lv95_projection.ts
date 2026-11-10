import {SubdivisionGranularityExpression, SubdivisionGranularitySetting} from '../../render/subdivision_granularity_settings.ts';
import {type PreparedShader, shaders} from '../../shaders/shaders.ts';
import {createTileMeshWithBuffers} from '../../util/create_tile_mesh.ts';
import {mercatorWorldCoordinateHelper} from '../mercator_coordinate.ts';

import type {Mesh} from '../../render/mesh.ts';
import type {Projection, TileMeshUsage} from './projection.ts';
import type {CanonicalTileID} from '../../tile/tile_id.ts';
import type {Context} from '../../webgl/context.ts';
import type {WorldCoordinateHelper} from '../transform_interface.ts';

export const LV95ShaderDefine = '#define PROJECTION_LV95';
export const LV95ShaderVariantKey = 'lv95';

/**
 * Straight edges of Web Mercator tiles curve in LV95 as much as they do on the globe, where the mercator scale
 * changes along them, so tiles are subdivided like the globe's.
 */
const granularitySettingsLV95: SubdivisionGranularitySetting = new SubdivisionGranularitySetting({
    fill: new SubdivisionGranularityExpression(128, 2),
    line: new SubdivisionGranularityExpression(512, 0),
    tile: new SubdivisionGranularityExpression(128, 4),
    // Stencil granularity must never be higher than fill granularity, or seams open between them.
    stencil: new SubdivisionGranularityExpression(128, 1),
    circle: 3
});

/**
 * The Swiss LV95 projection (EPSG:2056), selected with `projection: {type: 'lv95'}`. Tiles stay in the Web Mercator
 * grid sources are cut in, and the vertex shader reprojects them into the LV95 plane the camera of
 * {@link LV95Transform} moves in. The plane is the Swiss grid scaled to match Web Mercator at Bern, so zoom levels
 * show Switzerland at the scale they do in mercator.
 */
export class LV95Projection implements Projection {
    private readonly _meshes = new Map<string, Mesh>();

    get name(): 'lv95' {
        return 'lv95';
    }

    /** Tiles are cut in Web Mercator, so that is the plane sources, the terrain and queries see. */
    get worldCoordinateHelper(): WorldCoordinateHelper {
        return mercatorWorldCoordinateHelper;
    }

    get useSubdivision(): boolean {
        return true;
    }

    get shaderVariantName(): string {
        return LV95ShaderVariantKey;
    }

    get shaderDefine(): string {
        return LV95ShaderDefine;
    }

    get shaderPreludeCode(): PreparedShader {
        return shaders.projectionLV95;
    }

    get vertexShaderPreludeCode(): string {
        return shaders.projectionLV95.vertexSource;
    }

    get subdivisionGranularity(): SubdivisionGranularitySetting {
        return granularitySettingsLV95;
    }

    get transitionState(): number {
        return 0;
    }

    /** Polar caps are not drawn: the cylinder runs through the Arctic far from where LV95 is meant to be used. */
    getMeshFromTileID(context: Context, canonical: CanonicalTileID, hasBorder: boolean, _allowPoles: boolean, usage: TileMeshUsage): Mesh {
        const granularity = (usage === 'stencil' ? granularitySettingsLV95.stencil : granularitySettingsLV95.tile).getGranularityForZoomLevel(canonical.z);
        const key = `${granularity.toString(36)}_${hasBorder ? 'b' : ''}`;
        let mesh = this._meshes.get(key);
        if (!mesh) {
            mesh = createTileMeshWithBuffers(context, {granularity, generateBorders: hasBorder});
            this._meshes.set(key, mesh);
        }
        return mesh;
    }

    recalculate(): void {
        // Do nothing.
    }

    hasTransition(): boolean {
        return false;
    }

    destroy(): void {
        for (const mesh of this._meshes.values()) mesh.destroy();
        this._meshes.clear();
    }
}
