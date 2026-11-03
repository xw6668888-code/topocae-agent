# G11 fixed engineering tasks

Units: mm, N, MPa, tonne, s. All conditions receive identical instructions and the scope convention in SYSTEM_PROMPT.txt.

## 1. Angle Bracket.stp (CAE)

Assign 45 steel to the entire part and use a global mesh size of 2 mm. Fully constrain all faces of every counterbored hole. Apply a uniformly distributed force of 100 N to the inner walls of each obround hole in the negative global Y direction. Perform a linear-elastic static analysis with the stated material and loads.

STEP SHA-256: `25d25e8e0ae79ae6a084a22a5d42c19828b3be2a43bea8ad3e767b1714154195`

## 2. Bearing Housing.stp (CAD)

Change the radius of all blind holes with an existing radius of 4 mm to 3 mm. Preserve their axes and axial extent, and leave unrelated geometry unchanged.

STEP SHA-256: `68c756ce4df6ba506348ea958cae4c264f1c542062a0fb58ba9b8152e92de6b8`

## 3. Clevis.stp (CAE)

Assign 45 steel to the entire part and use a global mesh size of 1 mm. Refine all chamfer faces to 0.3 mm. Fully constrain the walls of all through holes with a radius of 3.765 mm. Apply a uniform directional surface traction of 50 MPa to the outward end face of the cylindrical boss facing the negative global Z direction in the positive global Z direction. Perform a linear-elastic static analysis with the stated material and loads.

STEP SHA-256: `0befb0c991dfbc9057dd644da22a4d99c0972374e1a98d8947d3da902fbd1f27`

## 4. Flanged Sleeve.stp (CAD)

Change the radius of all through holes with an existing radius of 20 mm to 15 mm. Preserve their axes and axial extent, and leave unrelated geometry unchanged.

STEP SHA-256: `af5669560aff44d525e0d49eae30fc5e9c9df5c142cf519ce330a5a00b83d034`

## 5. Flanged Support.stp (CAE)

Assign 45 steel to the entire part and use a global mesh size of 1 mm. Fully constrain all faces of every countersunk hole. Apply a uniformly distributed resultant force of 500 N to the wall of the through hole with a radius of 4 mm in the positive global Z direction. Perform a linear-elastic static analysis with the stated material and loads.

STEP SHA-256: `61690b63de49bffc204e83f6ae0ae4c7ea821753bda94b0f946eab570e4cb4d6`

## 6. Guide Rail.stp (CAD)

Reduce the adjacent center-to-center spacing of the three counterbored holes from 200 mm to 150 mm. Keep the central hole fixed and move the two outer holes toward it along the existing hole pattern. Preserve the hole geometry and unrelated features.

STEP SHA-256: `b2f942c6844f8eb737692dd8aa23d6273068868e050c2541294accf27fe01c03`

## 7. Mounting Plate.stp (CAD)

Change the radius of all through holes with an existing radius of 22.5 mm to 24 mm. Preserve their axes and axial extent, and leave unrelated geometry unchanged.

STEP SHA-256: `dccd20bb196660a6c1d5d52a813c2632c9d7291dec1712f8dc6b0aacecf6e14a`

## 8. Sealing Cover.stp (CAE)

Assign brass to the entire part and use a global mesh size of 5 mm. Refine all chamfer faces to 2 mm. Fully constrain all faces of every counterbored hole. Apply a uniform internal pressure of 10 MPa to all sidewalls and the bottom of the circular pocket. Perform a linear-elastic static analysis with the stated material and loads.

STEP SHA-256: `661e71aba1a73ebf8e86e73694b9f704124cd8ca08c1163097c31ecce76d1738`

## 9. Stiffened Panel.stp (CAE)

Assign 45 steel to the entire part and use a global mesh size of 10 mm. Fully constrain all faces of every counterbored hole. Apply a uniformly distributed force of 1000 N to the top face of each of the two rectangular bosses in the negative global Y direction. Perform a linear-elastic static analysis with the stated material and loads.

STEP SHA-256: `8de0640323efd148ff63de24f82cefc16612c24f44dbdfbb36096372204589a5`

## 10. Support Bracket.stp (CAD)

Change the thickness of both ribs from 2 mm to 4 mm by moving their two sides symmetrically about their existing midplanes. Keep the midplanes fixed and preserve unrelated geometry.

STEP SHA-256: `c61acff8bd576b87100d2366b3b3ce559ac9f03a01dd157c31486bbb50fc1a47`

## 11. Support Frame.stp (CAD)

Change the thickness of both webs from 4.6985 mm to 3 mm by moving their two sides symmetrically about their existing midplanes. Keep the midplanes fixed and preserve unrelated geometry.

STEP SHA-256: `79e3a204524f0de7284fd1318e5dd20353c3608b2a933eee12f83ff4ff655e70`

## 12. Turbine Blade.stp (CAE)

Assign plain carbon steel to the entire part and use a global mesh size of 1 mm. Fully constrain all faces of every slot_rect and slot_o feature. Apply a centrifugal body load to the entire part at an angular velocity of 100 rad/s about the axis through (0, 0, 0) mm with direction (0, 0, 1). Perform a linear-elastic static analysis with the stated material and loads.

STEP SHA-256: `c8444c78edafc1f70357ec4e08c340ee1b40218485a5057fcf3e6c97d98c41ee`

## 13. U-Shaped Bracket.stp (CAE)

Assign 45 steel to the entire part and use a global mesh size of 1 mm. Fully constrain the walls of all through holes. Apply a uniformly distributed force of 500 N to the outward end face of each cylindrical boss facing the positive global Y direction in the negative global Y direction. Perform a linear-elastic static analysis with the stated material and loads.

STEP SHA-256: `72356e514bcd4f5a955819ba9143842bd9298c45c3e4d65a8b62a88d903903bc`
