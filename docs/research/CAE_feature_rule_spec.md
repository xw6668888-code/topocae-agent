# CAE Feature Rule Specification

This document defines executable B-Rep rules for the first-paper scope:
single-part structural static preprocessing feature recognition and selectable
region extraction. The rules do not depend on CAD feature history. They use
STEP-derived faces, surface primitives, boundary edges, edge convexity,
adjacency, area, axis/normal, and shape descriptors.

## Scope

Target features are CAE-queryable regions, not complete manufacturing history:

- round through holes, blind holes, coaxial multi-radius holes, countersinks;
- obround holes;
- rectangular pockets and rectangular bosses;
- U/V/O slots;
- cylindrical bosses;
- ribs/webs as thin-wall candidates;
- fillets and chamfers as standalone transition targets.

Flanges, rails, lugs, and mounting plates are not first-paper target classes.
They may appear only as ordinary stock/background geometry.

## Common Graph Data

Each face node should expose:

- `primitive_surface`: `plane`, `cylinder`, `cone`, `toroid`, `bspline_surface`,
  `surface_of_revolution`, or `surface_of_extrusion`;
- `physical.area` and `physical.center_of_mass`;
- `specific.normal` for planes;
- `specific.axis` and `specific.radius` for cylindrical/conical/revolved
  surfaces when available;
- `shape_descriptors`: projected aspect ratio, projected long/short extents,
  UV span, bbox dimensions, edge count, and neighbor-scale descriptors.

Each graph edge should expose:

- curve type: line, circle, spline, or unknown;
- edge length;
- convexity: concave, convex, smooth, or unknown;
- dihedral angle when available.

## Rule Output

Rules produce candidate feature objects:

- `macro_feature`;
- `face_indices`;
- evidence strings;
- optional semantic subtypes used to create selectable regions.

Rules should prefer high recall at the candidate stage. False positives are
handled by candidate scoring, topology validation, and later model training.

## Transition Rules

### Fillet

Typical B-Rep form:

- one cylindrical, toroidal, surface-of-revolution, surface-of-extrusion, or
  spline transition face;
- adjacent to two or more larger faces;
- bounded by long line/circle/spline edges;
- usually smaller than parent stock faces and has a narrow swept width.

Executable candidate rule:

- accept a single curved face predicted as `stock`, `fillet`, `slot_o`, or
  an invalid transition-like class;
- require area above tiny-noise threshold;
- require at least two boundary neighbors;
- reject obvious holes when circular opening evidence indicates through-hole
  wall topology;
- prefer evidence from cylinder/revolved/extruded/toroidal primitive or
  small curved `bspline_surface`.

Selectable region:

- all transition faces.

### Chamfer

Typical B-Rep form:

- one narrow planar bevel face;
- high projected aspect ratio;
- adjacent to two larger parent faces;
- often connected by convex or smooth line edges;
- area much smaller than adjacent parent faces.

Executable candidate rule:

- accept a single plane predicted as `stock` or `chamfer`;
- require high projected aspect ratio or long-to-short extent ratio;
- require at least two neighboring faces through line edges;
- require the face area to be small relative to at least one adjacent parent
  face when neighbor areas are available.

Selectable region:

- bevel face.

## Slot Rules

### V Slot

Typical B-Rep form:

- two long inclined side planes;
- concave valley line between side faces, or two long inclined planes with
  mirrored normals when edge convexity is unavailable;
- long dimension much larger than section width.

Executable candidate rule:

- search stock/slot-like planes with high projected aspect ratio;
- pair adjacent planes with a concave line edge or inclined plane-pair evidence;
- validate with `slot_v` topology;
- output a two-face instance.

Selectable region:

- side faces as slot wall/load faces.

### O Slot

Typical B-Rep form:

- one long swept curved slot surface in simplified exports, often represented as
  `cylinder`, `surface_of_revolution`, `surface_of_extrusion`, or
  `bspline_surface`;
- long boundary edges and high area relative to ordinary fillets;
- may be produced by curve sweep, so it may not be a perfect cylinder.

Executable candidate rule:

- accept a stock curved/swept face with large area and at least two long
  boundary edges;
- accept `bspline_surface` when it has slot-scale area and neighboring line
  edges;
- remap to `slot_o` before generic fillet repair if it is slot-scale.

Selectable region:

- swept slot wall/bottom surface.

## Rib/Web Rule

Typical B-Rep form:

- a thin-wall group with two large near-parallel side planes;
- one top/narrow plane or sloped cap plane;
- thickness much smaller than length/height;
- attached to larger stock faces, normally through convex or smooth edges.

Executable candidate rule:

- search stock planes for near-parallel side-face pairs;
- require center distance divided by side short extent to be small;
- require long extent divided by thickness to be large;
- add an adjacent narrow cap/top plane when available;
- output a three-face `rib` instance when top plane exists, or a
  `rib_web_candidate` during candidate discovery.

Selectable regions:

- side faces for shell/thin-wall idealization;
- top face for auxiliary selection only.

## Hole And Pocket Rules

The current v4.x implementation already contains executable validation for:

- single cylindrical through holes;
- cylinder plus bottom blind holes;
- cone plus cylinder countersinks;
- coaxial multi-radius holes;
- rectangular pocket/boss completion;
- obround holes.

These should remain in the topology validation layer. The v4.4 priority is not
to replace them, but to add missing stock-face candidate discovery for
transition, slot, and rib features.

## Evaluation Rules

For simplified real STEP tests where face ids are unstable, use area-based
truth matching:

- macro feature class;
- feature-object face count;
- total feature area within tolerance.

For synthetic generated data, use full face-level labels and instance ids.
