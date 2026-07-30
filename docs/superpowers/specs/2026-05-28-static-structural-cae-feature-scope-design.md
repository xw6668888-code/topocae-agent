# Static Structural CAE Feature Scope Design

## Goal

This project version targets single-part static structural preprocessing. The dataset and pipeline should recognize local B-Rep face features that can be explicitly queried for finite element boundary conditions, loads, mesh preparation, or idealization, then map the selected face regions to Abaqus actions.

## Scope

The V1 feature scope is intentionally local and queryable. It includes round hole features, obround holes, slots, pockets, simple bosses, ribs/webs, fillets, chamfers, and derived stock-face roles. It excludes part-level or assembly-like concepts such as flange, bolt-hole pattern, bearing assembly semantics, contact, pretension, and full natural-language simulation intent.

Obround holes stay in the hole family because they are a common engineering hole feature. Hole groups are not a macro feature; they are inferred later from multiple single-hole feature objects by radius, axis, center spacing, and layout.

## Core Macro Features

The V1 generator set should produce these canonical macro features:

- `through_hole_round`
- `blind_hole_round`
- `stepped_hole_round`
- `counterbore_hole`
- `countersink_hole`
- `obround_hole`
- `slot_u`
- `slot_rect`
- `pocket_rect`
- `pocket_circular`
- `cylinder_boss`
- `rectangular_boss`
- `rib`
- `web`
- `fillet`
- `chamfer`

Derived CAE preparation labels remain separate from macro geometry labels:

- `thin_wall_region`
- `thickness_transition`
- `midsurface_candidate`
- `mesh_sensitive_region`
- `shell_candidate`

## Excluded Or Downgraded Features

The following labels may remain in the ontology for backward compatibility, but they are not enabled for the V1 paper dataset:

- `bolt_hole_pattern`: derived later from multiple hole objects, not trained as a geometry class.
- `flange`: too close to a part-level concept for the current single-part local-feature scope.
- `support_pad`: merged into `rectangular_boss`; support/load meaning belongs to an analysis case.
- `pad`: canonicalized as `rectangular_boss`.
- `cone_boss`, polygon/frustum bosses, triangular/rectangular cutouts, `notch_relief`, `bearing_seat`, and `gusset`: optional V2 material if a clear CAE query and robust generator are needed.

## Selectable Regions

The model should not output only a macro feature name. Each feature object should expose selectable face roles that can be used by structured simulation requests.

Examples:

- Blind hole: cylindrical wall, bottom face.
- Counterbore: large cylinder, shoulder plane, small cylinder.
- Obround hole: side planes, end cylinders.
- Rectangular boss: top face, side faces.
- Pocket: bottom face, side faces.
- Rib/web: side faces, top face, thin-wall candidate.
- Stock: top external face, bottom external face, side external face, largest planar face.

## Implementation Strategy

Keep existing generator modules to preserve prior work. Narrow the V1 path by changing default generation and coverage targets, plus ontology enabled flags and aliases. Do not delete non-core code in this phase.

The first implementation pass should:

1. Canonicalize `pad` output to `rectangular_boss`.
2. Keep `obround_hole` enabled in the hole family.
3. Disable part-level and non-core macro labels for default V1 coverage.
4. Restrict default generators to the V1 core set.
5. Make `BossFeatureGenerator` produce only cylindrical bosses for V1 stability.
6. Update derived CAE labels so thin-wall logic only depends on rib/web and rectangular-boss transition faces.
7. Add a lightweight regression test that verifies the V1 ontology/config scope.

## Success Criteria

- Default dataset generation no longer includes flange, bolt-hole pattern, support pad, bearing seat, gusset, notch relief, polygon cutouts, or non-core polygon/frustum bosses.
- `pad` generator output resolves to `rectangular_boss`.
- `obround_hole` remains enabled and in the hole family.
- Coverage targets are limited to enabled V1 core features.
- Derived thin-wall and mesh-sensitive tags still work for rib/web and rectangular-boss transition regions.
