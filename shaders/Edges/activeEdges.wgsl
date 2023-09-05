struct Vertex {
    position: vec3f,
    value: f32,
}

@group(0) @binding(0)
var <storage, read> vertices : array<Vertex>;
@group(0) @binding(1)
var <storage, read_write> edgesId : array<u32>;
@group(0) @binding(2)
var<storage, read_write> chunkEdges : array<u32>;
@group(0) @binding(3)
var<storage, read_write> edgeChunkOffset : array<u32>;
@group(0) @binding(4)
var<storage, read_write> edgeMid : array<vec3f>;
@group(0) @binding(5)
var<storage, read_write> rawTri : array<u32>;
@group(0) @binding(6)
var<storage, read_write> indexCount : array<u32>;
@group(0) @binding(7)
var<storage, read_write> rawTriOffset : array<u32>;
// @group(1) @binding(9)
// var<storage, read_write> indexBuffer : array<u32>;

const INVALID = 0u - 1u;
const inv = INVALID;
const Min : vec3f = vec3f(-1.0,-1.0,-1.0);
const Max : vec3f = vec3f(1.0,1.0,1.0);
override STEP : f32 = 2.0 / 64.0;
override X : u32 = 64u;
override Y : u32 = 64u;
override Z : u32 = 64u;
override NbVertices = (X+1) * (Y+1) * (Z+1);
override CHUNKSIZE = 64u;
//const step : vec3f = vec3f

fn computeVertexId(globalId: u32) -> vec3u {
    var id = globalId;
    let z = id / ((X+1)*(Y+1));
    id %= ((X+1)*(Y+1));
    let y = id / (X+1);
    let x = id % (X+1);
    return vec3u(x, y, z);
}

fn computeVertexValue(pos: vec3f) -> f32 {
    // www-sop.inria.fr/galaad/surface/
    // chubs
    return pow(pos.x, 4)+pow(pos.y, 4)+pow(pos.z, 4)-pow(pos.x, 2)-pow(pos.y, 2)-pow(pos.z, 2)+0.5; 
}

fn vertexId(coord: vec3u) -> u32{
    return coord.x + (X+1) * coord.y + (X+1)*(Y+1) * coord.z;
}

// fn edgeId(coord: vec3u) -> u32 {

// }


override yEOff = X*(Y+1)*(Z+1);
override zEOff = yEOff+(X+1)*Y*(Z+1);
override xEOff = X*(Y+1);
override xEOff1 = (X+1)*(Y+1);
override yVOff = (X + 1);
override zVOff = yVOff * (Y + 1);
fn cellCoord(id: u32) -> vec3u {
    var coord : vec3u;
    coord.z = id / (X * Y);
    var id2 = id % (X*Y); 
    coord.y = id2 / X;
    coord.x = id2 % X;
    return coord;
}

fn cellEdge(coord: vec3u, i: u32) -> u32 {
    var e = INVALID;
    switch(i) {
        case 0: { e = coord.x + X * coord.y + xEOff * coord.z; }
        case 1: { e = coord.x + X * (coord.y+1) + xEOff * coord.z; }
        case 2: { e = coord.x + X * coord.y + xEOff*(coord.z+1); }
        case 3: { e = coord.x + X * (coord.y+1) + xEOff*(coord.z+1);}

        case 4: { e = yEOff + coord.x + (X+1)*coord.y+ xEOff * coord.z; }
        case 5: { e = yEOff + (coord.x+1) + (X+1)*coord.y+ xEOff * coord.z; }
        case 6: { e = yEOff + coord.x + (X+1)*coord.y+ xEOff * (coord.z+1); }
        case 7: { e = yEOff + (coord.x+1) + (X+1)*coord.y+ xEOff * (coord.z+1);}

        case 8: { e = zEOff + coord.x + (X+1)*coord.y + xEOff1 * coord.z; }
        case 9: { e = zEOff + (coord.x+1) + (X+1)*coord.y + xEOff1 * coord.z; }
        case 10: { e = zEOff + coord.x + (X+1)*(coord.y+1) + xEOff1 * coord.z; }
        case 11: { e = zEOff + (coord.x+1) + (X+1)*(coord.y+1) + xEOff1 * coord.z; }
        default: {}
    }
    return e;
}

override nbX = (X)*(Y+1)*(Z+1);
override nbY = (X+1)*(Y)*(Z+1);
override nbZ = (X+1)*(Y+1)*(Z);

fn edgeCell(e: u32) -> vec3u {
    let ABC = vec3u(
        u32(e >= nbX) * nbX + u32(e >= (nbX + nbY)) * nbY,
        (X + u32(e > nbX)) * (Y + u32((e < nbX) || (e > nbX + nbY))),
        X + u32(e > nbX)
    );

    var xyz : vec3u;
    var e_ = e - ABC.x;
    xyz.z = e_ / ABC.y;
    e_ = e_ % ABC.y;
    xyz.y = e_ / ABC.z;
    xyz.x = e_ % ABC.z;

    return xyz;
}

@compute @workgroup_size(64)
fn markActiveEdges(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&edgesId)) {
        return;
    }

    let cell = edgeCell(global_id.x);

    let axis = vec3u(
        u32(global_id.x < nbX),
        u32((global_id.x < nbY + nbX) && global_id.x >= nbX),
        u32(global_id.x >= (nbX + nbY))
    );

    let v0 = vertices[vertexId(cell)].value;
    let v1 = vertices[vertexId(cell+axis)].value;

    if((v0 * v1) <= 0.0) {
        edgesId[global_id.x] = global_id.x; 
    } else {
        edgesId[global_id.x] = INVALID;
    }
}

@compute @workgroup_size(64)
fn countEdgesPerChunk(@builtin(global_invocation_id) global_id : vec3<u32>) {
    _ = X;
    _ = Y;
    _ = Z;
	if(global_id.x >= arrayLength(&chunkEdges) - 1) {
        return;
    }


    /// first edge of chunk
    let chunkOffset = global_id.x * CHUNKSIZE;
    var localOffset = 0u;
    for(var i = 0u; i < CHUNKSIZE; i++) {
        if(edgesId[chunkOffset + i] != INVALID) {
            edgeChunkOffset[chunkOffset + i] = localOffset;
            localOffset++;
        } else {
            edgeChunkOffset[chunkOffset + i] = INVALID;
        }
    }

    chunkEdges[global_id.x] = localOffset;
}

@compute @workgroup_size(1)
fn reduceEdgeCount() {
    _ = X;
    _ = Y;
    _ = Z;
    _ = CHUNKSIZE;

    var off0 = chunkEdges[0];
    var off1 = 0u;
    chunkEdges[0] = 0;
    for(var i = 0u; i < arrayLength(&chunkEdges) - 1; i++) {
        off1 = chunkEdges[i+1];
        chunkEdges[i+1] = chunkEdges[i] + off0;
        off0 = off1;
    }
}

@compute @workgroup_size(64)
fn completeEdgeOffsets(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&edgesId)
        || edgesId[global_id.x] == INVALID) {
        return;
    }

    let chunkId = global_id.x / CHUNKSIZE;
    let chunkOffset = chunkEdges[chunkId];

    edgeChunkOffset[global_id.x] += chunkOffset ;
}

@compute @workgroup_size(64)
fn computEdgeMid(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&edgesId)
        || edgesId[global_id.x] == INVALID) {
        return;
    }

    let cell = edgeCell(global_id.x);

    let axis = vec3u(
        u32(global_id.x < nbX),
        u32((global_id.x < nbY + nbX) && global_id.x >= nbX),
        u32(global_id.x >= (nbX + nbY))
    );

    
    let v0 = vertices[vertexId(cell)];
    let v1 = vertices[vertexId(cell+axis)];
    let dist = abs(v0.value) + abs(v1.value);

    let offset = edgeChunkOffset[global_id.x];
    edgeMid[offset] = mix(v0.position, v1.position, abs(v0.value)/dist);
    // edgeMid[offset] = (v0.position + v1.position) / 2.0;
}



const triTable = array<u32, 4096>(
/*0*/ inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*1*/ 0, 8, 4, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*2*/ 0, 5, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*3*/ 8, 4, 9, 9, 4, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*4*/ 1, 4, 10, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*5*/ 10, 1, 8, 8, 1, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*6*/ 1, 4, 10, 0, 5, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*7*/ 9, 1, 5, 9, 10, 1, 9, 8, 10, inv, inv, inv, inv, inv, inv, inv,
/*8*/ 5, 1, 11, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*9*/ 5, 1, 11, 4, 0, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*10*/ 1, 11, 0, 0, 11, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*11*/ 11, 4, 1, 11, 8, 4, 11, 9, 8, inv, inv, inv, inv, inv, inv, inv,
/*12*/ 4, 10, 5, 5, 10, 11, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*13*/ 8, 5, 0, 8, 11, 5, 8, 10, 11, inv, inv, inv, inv, inv, inv, inv,
/*14*/ 10, 0, 4, 10, 9, 0, 10, 11, 9, inv, inv, inv, inv, inv, inv, inv,
/*15*/ 10, 11, 8, 8, 11, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*16*/ 2, 6, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*17*/ 4, 0, 6, 6, 0, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*18*/ 8, 2, 6, 9, 0, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*19*/ 6, 9, 2, 6, 5, 9, 6, 4, 5, inv, inv, inv, inv, inv, inv, inv,
/*20*/ 4, 10, 1, 6, 8, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*21*/ 1, 6, 10, 1, 2, 6, 1, 0, 2, inv, inv, inv, inv, inv, inv, inv,
/*22*/ 0, 5, 9, 4, 10, 1, 6, 8, 2, inv, inv, inv, inv, inv, inv, inv,
/*23*/ 10, 1, 6, 6, 1, 2, 2, 1, 5, 2, 5, 9, inv, inv, inv, inv,
/*24*/ 5, 1, 11, 6, 8, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*25*/ 2, 6, 0, 0, 6, 4, 11, 5, 1, inv, inv, inv, inv, inv, inv, inv,
/*26*/ 1, 11, 0, 0, 11, 9, 6, 8, 2, inv, inv, inv, inv, inv, inv, inv,
/*27*/ 11, 9, 1, 9, 2, 6, 1, 9, 6, 1, 6, 4, inv, inv, inv, inv,
/*28*/ 11, 5, 10, 10, 5, 4, 2, 6, 8, inv, inv, inv, inv, inv, inv, inv,
/*29*/ 2, 6, 10, 2, 10, 5, 10, 11, 5, 0, 2, 5, inv, inv, inv, inv,
/*30*/ 2, 6, 8, 10, 0, 4, 10, 9, 0, 10, 11, 9, inv, inv, inv, inv,
/*31*/ 2, 6, 9, 6, 10, 9, 10, 11, 9, inv, inv, inv, inv, inv, inv, inv,
/*32*/ 9, 7, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*33*/ 0, 8, 4, 2, 9, 7, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*34*/ 7, 2, 5, 5, 2, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*35*/ 4, 2, 8, 4, 7, 2, 4, 5, 7, inv, inv, inv, inv, inv, inv, inv,
/*36*/ 1, 4, 10, 2, 9, 7, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*37*/ 10, 1, 8, 8, 1, 0, 7, 2, 9, inv, inv, inv, inv, inv, inv, inv,
/*38*/ 7, 2, 5, 5, 2, 0, 10, 1, 4, inv, inv, inv, inv, inv, inv, inv,
/*39*/ 7, 2, 8, 7, 8, 1, 8, 10, 1, 5, 7, 1, inv, inv, inv, inv,
/*40*/ 9, 7, 2, 11, 5, 1, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*41*/ 5, 1, 11, 0, 8, 4, 2, 9, 7, inv, inv, inv, inv, inv, inv, inv,
/*42*/ 2, 11, 7, 2, 1, 11, 2, 0, 1, inv, inv, inv, inv, inv, inv, inv,
/*43*/ 7, 2, 11, 11, 2, 1, 1, 2, 8, 1, 8, 4, inv, inv, inv, inv,
/*44*/ 4, 10, 5, 5, 10, 11, 2, 9, 7, inv, inv, inv, inv, inv, inv, inv,
/*45*/ 7, 2, 9, 8, 5, 0, 8, 11, 5, 8, 10, 11, inv, inv, inv, inv,
/*46*/ 10, 11, 4, 11, 7, 2, 4, 11, 2, 4, 2, 0, inv, inv, inv, inv,
/*47*/ 7, 2, 11, 2, 8, 11, 8, 10, 11, inv, inv, inv, inv, inv, inv, inv,
/*48*/ 6, 8, 7, 7, 8, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*49*/ 7, 0, 9, 7, 4, 0, 7, 6, 4, inv, inv, inv, inv, inv, inv, inv,
/*50*/ 5, 8, 0, 5, 6, 8, 5, 7, 6, inv, inv, inv, inv, inv, inv, inv,
/*51*/ 4, 5, 6, 6, 5, 7, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*52*/ 9, 7, 8, 8, 7, 6, 1, 4, 10, inv, inv, inv, inv, inv, inv, inv,
/*53*/ 7, 6, 9, 6, 10, 1, 9, 6, 1, 9, 1, 0, inv, inv, inv, inv,
/*54*/ 10, 1, 4, 5, 8, 0, 5, 6, 8, 5, 7, 6, inv, inv, inv, inv,
/*55*/ 10, 1, 6, 1, 5, 6, 5, 7, 6, inv, inv, inv, inv, inv, inv, inv,
/*56*/ 6, 8, 7, 7, 8, 9, 1, 11, 5, inv, inv, inv, inv, inv, inv, inv,
/*57*/ 1, 11, 5, 7, 0, 9, 7, 4, 0, 7, 6, 4, inv, inv, inv, inv,
/*58*/ 6, 8, 0, 6, 0, 11, 0, 1, 11, 7, 6, 11, inv, inv, inv, inv,
/*59*/ 1, 11, 4, 11, 7, 4, 7, 6, 4, inv, inv, inv, inv, inv, inv, inv,
/*60*/ 4, 10, 11, 4, 11, 5, 6, 8, 7, 8, 9, 7, inv, inv, inv, inv,
/*61*/ 11, 6, 10, 7, 6, 11, 9, 5, 0, inv, inv, inv, inv, inv, inv, inv,
/*62*/ 6, 11, 7, 10, 11, 6, 4, 8, 0, inv, inv, inv, inv, inv, inv, inv,
/*63*/ 11, 6, 10, 7, 6, 11, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*64*/ 3, 10, 6, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*65*/ 10, 6, 3, 8, 4, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*66*/ 3, 10, 6, 0, 5, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*67*/ 5, 9, 4, 4, 9, 8, 3, 10, 6, inv, inv, inv, inv, inv, inv, inv,
/*68*/ 6, 3, 4, 4, 3, 1, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*69*/ 3, 8, 6, 3, 0, 8, 3, 1, 0, inv, inv, inv, inv, inv, inv, inv,
/*70*/ 6, 3, 4, 4, 3, 1, 9, 0, 5, inv, inv, inv, inv, inv, inv, inv,
/*71*/ 9, 8, 5, 8, 6, 3, 5, 8, 3, 5, 3, 1, inv, inv, inv, inv,
/*72*/ 3, 10, 6, 1, 11, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*73*/ 1, 11, 5, 10, 6, 3, 8, 4, 0, inv, inv, inv, inv, inv, inv, inv,
/*74*/ 9, 0, 11, 11, 0, 1, 6, 3, 10, inv, inv, inv, inv, inv, inv, inv,
/*75*/ 6, 3, 10, 11, 4, 1, 11, 8, 4, 11, 9, 8, inv, inv, inv, inv,
/*76*/ 5, 3, 11, 5, 6, 3, 5, 4, 6, inv, inv, inv, inv, inv, inv, inv,
/*77*/ 6, 3, 8, 8, 3, 0, 0, 3, 11, 0, 11, 5, inv, inv, inv, inv,
/*78*/ 6, 3, 11, 6, 11, 0, 11, 9, 0, 4, 6, 0, inv, inv, inv, inv,
/*79*/ 6, 3, 8, 3, 11, 8, 11, 9, 8, inv, inv, inv, inv, inv, inv, inv,
/*80*/ 8, 2, 10, 10, 2, 3, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*81*/ 0, 10, 4, 0, 3, 10, 0, 2, 3, inv, inv, inv, inv, inv, inv, inv,
/*82*/ 3, 10, 2, 2, 10, 8, 5, 9, 0, inv, inv, inv, inv, inv, inv, inv,
/*83*/ 3, 10, 4, 3, 4, 9, 4, 5, 9, 2, 3, 9, inv, inv, inv, inv,
/*84*/ 2, 4, 8, 2, 1, 4, 2, 3, 1, inv, inv, inv, inv, inv, inv, inv,
/*85*/ 1, 0, 3, 3, 0, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*86*/ 5, 9, 0, 2, 4, 8, 2, 1, 4, 2, 3, 1, inv, inv, inv, inv,
/*87*/ 5, 9, 1, 9, 2, 1, 2, 3, 1, inv, inv, inv, inv, inv, inv, inv,
/*88*/ 8, 2, 10, 10, 2, 3, 5, 1, 11, inv, inv, inv, inv, inv, inv, inv,
/*89*/ 11, 5, 1, 0, 10, 4, 0, 3, 10, 0, 2, 3, inv, inv, inv, inv,
/*90*/ 8, 2, 3, 8, 3, 10, 9, 0, 11, 0, 1, 11, inv, inv, inv, inv,
/*91*/ 3, 9, 2, 11, 9, 3, 1, 10, 4, inv, inv, inv, inv, inv, inv, inv,
/*92*/ 5, 4, 11, 4, 8, 2, 11, 4, 2, 11, 2, 3, inv, inv, inv, inv,
/*93*/ 11, 5, 3, 5, 0, 3, 0, 2, 3, inv, inv, inv, inv, inv, inv, inv,
/*94*/ 9, 3, 11, 2, 3, 9, 8, 0, 4, inv, inv, inv, inv, inv, inv, inv,
/*95*/ 9, 3, 11, 2, 3, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*96*/ 6, 3, 10, 7, 2, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*97*/ 8, 4, 0, 6, 3, 10, 7, 2, 9, inv, inv, inv, inv, inv, inv, inv,
/*98*/ 0, 5, 2, 2, 5, 7, 10, 6, 3, inv, inv, inv, inv, inv, inv, inv,
/*99*/ 3, 10, 6, 4, 2, 8, 4, 7, 2, 4, 5, 7, inv, inv, inv, inv,
/*100*/ 1, 4, 3, 3, 4, 6, 9, 7, 2, inv, inv, inv, inv, inv, inv, inv,
/*101*/ 9, 7, 2, 3, 8, 6, 3, 0, 8, 3, 1, 0, inv, inv, inv, inv,
/*102*/ 6, 3, 1, 6, 1, 4, 7, 2, 5, 2, 0, 5, inv, inv, inv, inv,
/*103*/ 7, 1, 5, 3, 1, 7, 6, 2, 8, inv, inv, inv, inv, inv, inv, inv,
/*104*/ 7, 2, 9, 3, 10, 6, 1, 11, 5, inv, inv, inv, inv, inv, inv, inv,
/*105*/ 0, 8, 4, 1, 11, 5, 6, 3, 10, 7, 2, 9, inv, inv, inv, inv,
/*106*/ 10, 6, 3, 2, 11, 7, 2, 1, 11, 2, 0, 1, inv, inv, inv, inv,
/*107*/ 4, 1, 10, 11, 7, 3, 8, 6, 2, inv, inv, inv, inv, inv, inv, inv,
/*108*/ 2, 9, 7, 5, 3, 11, 5, 6, 3, 5, 4, 6, inv, inv, inv, inv,
/*109*/ 5, 0, 9, 8, 6, 2, 11, 7, 3, inv, inv, inv, inv, inv, inv, inv,
/*110*/ 6, 0, 4, 2, 0, 6, 7, 3, 11, inv, inv, inv, inv, inv, inv, inv,
/*111*/ 8, 6, 2, 11, 7, 3, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*112*/ 10, 7, 3, 10, 9, 7, 10, 8, 9, inv, inv, inv, inv, inv, inv, inv,
/*113*/ 4, 0, 10, 10, 0, 3, 3, 0, 9, 3, 9, 7, inv, inv, inv, inv,
/*114*/ 10, 8, 3, 8, 0, 5, 3, 8, 5, 3, 5, 7, inv, inv, inv, inv,
/*115*/ 3, 10, 7, 10, 4, 7, 4, 5, 7, inv, inv, inv, inv, inv, inv, inv,
/*116*/ 1, 4, 8, 1, 8, 7, 8, 9, 7, 3, 1, 7, inv, inv, inv, inv,
/*117*/ 9, 7, 0, 7, 3, 0, 3, 1, 0, inv, inv, inv, inv, inv, inv, inv,
/*118*/ 1, 7, 3, 5, 7, 1, 0, 4, 8, inv, inv, inv, inv, inv, inv, inv,
/*119*/ 1, 7, 3, 5, 7, 1, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*120*/ 5, 1, 11, 10, 7, 3, 10, 9, 7, 10, 8, 9, inv, inv, inv, inv,
/*121*/ 7, 3, 11, 10, 4, 1, 9, 5, 0, inv, inv, inv, inv, inv, inv, inv,
/*122*/ 1, 8, 0, 10, 8, 1, 3, 11, 7, inv, inv, inv, inv, inv, inv, inv,
/*123*/ 7, 3, 11, 4, 1, 10, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*124*/ 9, 4, 8, 5, 4, 9, 11, 7, 3, inv, inv, inv, inv, inv, inv, inv,
/*125*/ 3, 11, 7, 0, 9, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*126*/ 4, 8, 0, 11, 7, 3, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*127*/ 3, 11, 7, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*128*/ 7, 11, 3, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*129*/ 0, 8, 4, 3, 7, 11, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*130*/ 7, 11, 3, 5, 9, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*131*/ 8, 4, 9, 9, 4, 5, 3, 7, 11, inv, inv, inv, inv, inv, inv, inv,
/*132*/ 11, 3, 7, 10, 1, 4, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*133*/ 0, 8, 1, 1, 8, 10, 7, 11, 3, inv, inv, inv, inv, inv, inv, inv,
/*134*/ 11, 3, 7, 1, 4, 10, 0, 5, 9, inv, inv, inv, inv, inv, inv, inv,
/*135*/ 3, 7, 11, 9, 1, 5, 9, 10, 1, 9, 8, 10, inv, inv, inv, inv,
/*136*/ 3, 7, 1, 1, 7, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*137*/ 3, 7, 1, 1, 7, 5, 8, 4, 0, inv, inv, inv, inv, inv, inv, inv,
/*138*/ 0, 7, 9, 0, 3, 7, 0, 1, 3, inv, inv, inv, inv, inv, inv, inv,
/*139*/ 3, 7, 9, 3, 9, 4, 9, 8, 4, 1, 3, 4, inv, inv, inv, inv,
/*140*/ 7, 10, 3, 7, 4, 10, 7, 5, 4, inv, inv, inv, inv, inv, inv, inv,
/*141*/ 8, 10, 0, 10, 3, 7, 0, 10, 7, 0, 7, 5, inv, inv, inv, inv,
/*142*/ 3, 7, 10, 10, 7, 4, 4, 7, 9, 4, 9, 0, inv, inv, inv, inv,
/*143*/ 3, 7, 10, 7, 9, 10, 9, 8, 10, inv, inv, inv, inv, inv, inv, inv,
/*144*/ 2, 6, 8, 3, 7, 11, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*145*/ 4, 0, 6, 6, 0, 2, 11, 3, 7, inv, inv, inv, inv, inv, inv, inv,
/*146*/ 9, 0, 5, 2, 6, 8, 3, 7, 11, inv, inv, inv, inv, inv, inv, inv,
/*147*/ 11, 3, 7, 6, 9, 2, 6, 5, 9, 6, 4, 5, inv, inv, inv, inv,
/*148*/ 10, 1, 4, 3, 7, 11, 2, 6, 8, inv, inv, inv, inv, inv, inv, inv,
/*149*/ 7, 11, 3, 1, 6, 10, 1, 2, 6, 1, 0, 2, inv, inv, inv, inv,
/*150*/ 1, 4, 10, 3, 7, 11, 8, 2, 6, 9, 0, 5, inv, inv, inv, inv,
/*151*/ 9, 2, 7, 6, 10, 3, 5, 11, 1, inv, inv, inv, inv, inv, inv, inv,
/*152*/ 5, 1, 7, 7, 1, 3, 8, 2, 6, inv, inv, inv, inv, inv, inv, inv,
/*153*/ 2, 6, 4, 2, 4, 0, 3, 7, 1, 7, 5, 1, inv, inv, inv, inv,
/*154*/ 6, 8, 2, 0, 7, 9, 0, 3, 7, 0, 1, 3, inv, inv, inv, inv,
/*155*/ 3, 4, 1, 6, 4, 3, 2, 7, 9, inv, inv, inv, inv, inv, inv, inv,
/*156*/ 8, 2, 6, 7, 10, 3, 7, 4, 10, 7, 5, 4, inv, inv, inv, inv,
/*157*/ 2, 5, 0, 7, 5, 2, 3, 6, 10, inv, inv, inv, inv, inv, inv, inv,
/*158*/ 0, 4, 8, 10, 3, 6, 9, 2, 7, inv, inv, inv, inv, inv, inv, inv,
/*159*/ 10, 3, 6, 9, 2, 7, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*160*/ 11, 3, 9, 9, 3, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*161*/ 11, 3, 9, 9, 3, 2, 4, 0, 8, inv, inv, inv, inv, inv, inv, inv,
/*162*/ 3, 5, 11, 3, 0, 5, 3, 2, 0, inv, inv, inv, inv, inv, inv, inv,
/*163*/ 3, 2, 11, 2, 8, 4, 11, 2, 4, 11, 4, 5, inv, inv, inv, inv,
/*164*/ 2, 9, 3, 3, 9, 11, 4, 10, 1, inv, inv, inv, inv, inv, inv, inv,
/*165*/ 0, 8, 10, 0, 10, 1, 2, 9, 3, 9, 11, 3, inv, inv, inv, inv,
/*166*/ 4, 10, 1, 3, 5, 11, 3, 0, 5, 3, 2, 0, inv, inv, inv, inv,
/*167*/ 10, 2, 8, 3, 2, 10, 11, 1, 5, inv, inv, inv, inv, inv, inv, inv,
/*168*/ 1, 9, 5, 1, 2, 9, 1, 3, 2, inv, inv, inv, inv, inv, inv, inv,
/*169*/ 8, 4, 0, 1, 9, 5, 1, 2, 9, 1, 3, 2, inv, inv, inv, inv,
/*170*/ 3, 2, 1, 1, 2, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*171*/ 8, 4, 2, 4, 1, 2, 1, 3, 2, inv, inv, inv, inv, inv, inv, inv,
/*172*/ 2, 9, 5, 2, 5, 10, 5, 4, 10, 3, 2, 10, inv, inv, inv, inv,
/*173*/ 2, 10, 3, 8, 10, 2, 0, 9, 5, inv, inv, inv, inv, inv, inv, inv,
/*174*/ 4, 10, 0, 10, 3, 0, 3, 2, 0, inv, inv, inv, inv, inv, inv, inv,
/*175*/ 10, 2, 8, 3, 2, 10, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*176*/ 8, 3, 6, 8, 11, 3, 8, 9, 11, inv, inv, inv, inv, inv, inv, inv,
/*177*/ 4, 0, 9, 4, 9, 3, 9, 11, 3, 6, 4, 3, inv, inv, inv, inv,
/*178*/ 6, 8, 3, 3, 8, 11, 11, 8, 0, 11, 0, 5, inv, inv, inv, inv,
/*179*/ 11, 3, 5, 3, 6, 5, 6, 4, 5, inv, inv, inv, inv, inv, inv, inv,
/*180*/ 1, 4, 10, 8, 3, 6, 8, 11, 3, 8, 9, 11, inv, inv, inv, inv,
/*181*/ 11, 0, 9, 1, 0, 11, 10, 3, 6, inv, inv, inv, inv, inv, inv, inv,
/*182*/ 5, 11, 1, 3, 6, 10, 0, 4, 8, inv, inv, inv, inv, inv, inv, inv,
/*183*/ 6, 10, 3, 5, 11, 1, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*184*/ 1, 3, 5, 3, 6, 8, 5, 3, 8, 5, 8, 9, inv, inv, inv, inv,
/*185*/ 4, 3, 6, 1, 3, 4, 5, 0, 9, inv, inv, inv, inv, inv, inv, inv,
/*186*/ 6, 8, 3, 8, 0, 3, 0, 1, 3, inv, inv, inv, inv, inv, inv, inv,
/*187*/ 4, 3, 6, 1, 3, 4, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*188*/ 4, 9, 5, 8, 9, 4, 6, 10, 3, inv, inv, inv, inv, inv, inv, inv,
/*189*/ 6, 10, 3, 9, 5, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*190*/ 3, 6, 10, 0, 4, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*191*/ 6, 10, 3, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*192*/ 10, 6, 11, 11, 6, 7, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*193*/ 7, 11, 6, 6, 11, 10, 0, 8, 4, inv, inv, inv, inv, inv, inv, inv,
/*194*/ 10, 6, 11, 11, 6, 7, 0, 5, 9, inv, inv, inv, inv, inv, inv, inv,
/*195*/ 10, 6, 7, 10, 7, 11, 8, 4, 9, 4, 5, 9, inv, inv, inv, inv,
/*196*/ 4, 11, 1, 4, 7, 11, 4, 6, 7, inv, inv, inv, inv, inv, inv, inv,
/*197*/ 0, 8, 6, 0, 6, 11, 6, 7, 11, 1, 0, 11, inv, inv, inv, inv,
/*198*/ 9, 0, 5, 4, 11, 1, 4, 7, 11, 4, 6, 7, inv, inv, inv, inv,
/*199*/ 7, 8, 6, 9, 8, 7, 5, 11, 1, inv, inv, inv, inv, inv, inv, inv,
/*200*/ 6, 1, 10, 6, 5, 1, 6, 7, 5, inv, inv, inv, inv, inv, inv, inv,
/*201*/ 0, 8, 4, 6, 1, 10, 6, 5, 1, 6, 7, 5, inv, inv, inv, inv,
/*202*/ 6, 7, 10, 7, 9, 0, 10, 7, 0, 10, 0, 1, inv, inv, inv, inv,
/*203*/ 8, 7, 9, 6, 7, 8, 10, 4, 1, inv, inv, inv, inv, inv, inv, inv,
/*204*/ 6, 7, 4, 4, 7, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*205*/ 0, 8, 5, 8, 6, 5, 6, 7, 5, inv, inv, inv, inv, inv, inv, inv,
/*206*/ 9, 0, 7, 0, 4, 7, 4, 6, 7, inv, inv, inv, inv, inv, inv, inv,
/*207*/ 7, 8, 6, 9, 8, 7, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*208*/ 11, 2, 7, 11, 8, 2, 11, 10, 8, inv, inv, inv, inv, inv, inv, inv,
/*209*/ 11, 10, 7, 10, 4, 0, 7, 10, 0, 7, 0, 2, inv, inv, inv, inv,
/*210*/ 0, 5, 9, 11, 2, 7, 11, 8, 2, 11, 10, 8, inv, inv, inv, inv,
/*211*/ 5, 10, 4, 11, 10, 5, 7, 9, 2, inv, inv, inv, inv, inv, inv, inv,
/*212*/ 8, 2, 4, 4, 2, 1, 1, 2, 7, 1, 7, 11, inv, inv, inv, inv,
/*213*/ 7, 11, 2, 11, 1, 2, 1, 0, 2, inv, inv, inv, inv, inv, inv, inv,
/*214*/ 11, 1, 5, 4, 8, 0, 7, 9, 2, inv, inv, inv, inv, inv, inv, inv,
/*215*/ 2, 7, 9, 1, 5, 11, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*216*/ 8, 2, 7, 8, 7, 1, 7, 5, 1, 10, 8, 1, inv, inv, inv, inv,
/*217*/ 5, 2, 7, 0, 2, 5, 4, 1, 10, inv, inv, inv, inv, inv, inv, inv,
/*218*/ 8, 1, 10, 0, 1, 8, 9, 2, 7, inv, inv, inv, inv, inv, inv, inv,
/*219*/ 10, 4, 1, 7, 9, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*220*/ 8, 2, 4, 2, 7, 4, 7, 5, 4, inv, inv, inv, inv, inv, inv, inv,
/*221*/ 5, 2, 7, 0, 2, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*222*/ 4, 8, 0, 7, 9, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*223*/ 2, 7, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*224*/ 9, 6, 2, 9, 10, 6, 9, 11, 10, inv, inv, inv, inv, inv, inv, inv,
/*225*/ 4, 0, 8, 9, 6, 2, 9, 10, 6, 9, 11, 10, inv, inv, inv, inv,
/*226*/ 10, 6, 2, 10, 2, 5, 2, 0, 5, 11, 10, 5, inv, inv, inv, inv,
/*227*/ 10, 5, 11, 4, 5, 10, 8, 6, 2, inv, inv, inv, inv, inv, inv, inv,
/*228*/ 4, 6, 1, 6, 2, 9, 1, 6, 9, 1, 9, 11, inv, inv, inv, inv,
/*229*/ 0, 11, 1, 9, 11, 0, 2, 8, 6, inv, inv, inv, inv, inv, inv, inv,
/*230*/ 0, 6, 2, 4, 6, 0, 1, 5, 11, inv, inv, inv, inv, inv, inv, inv,
/*231*/ 11, 1, 5, 2, 8, 6, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*232*/ 10, 6, 1, 1, 6, 5, 5, 6, 2, 5, 2, 9, inv, inv, inv, inv,
/*233*/ 9, 5, 0, 1, 10, 4, 2, 8, 6, inv, inv, inv, inv, inv, inv, inv,
/*234*/ 10, 6, 1, 6, 2, 1, 2, 0, 1, inv, inv, inv, inv, inv, inv, inv,
/*235*/ 1, 10, 4, 2, 8, 6, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*236*/ 2, 9, 6, 9, 5, 6, 5, 4, 6, inv, inv, inv, inv, inv, inv, inv,
/*237*/ 6, 2, 8, 5, 0, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*238*/ 6, 0, 4, 2, 0, 6, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*239*/ 8, 6, 2, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*240*/ 8, 9, 10, 10, 9, 11, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*241*/ 4, 0, 10, 0, 9, 10, 9, 11, 10, inv, inv, inv, inv, inv, inv, inv,
/*242*/ 0, 5, 8, 5, 11, 8, 11, 10, 8, inv, inv, inv, inv, inv, inv, inv,
/*243*/ 5, 10, 4, 11, 10, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*244*/ 1, 4, 11, 4, 8, 11, 8, 9, 11, inv, inv, inv, inv, inv, inv, inv,
/*245*/ 0, 11, 1, 9, 11, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*246*/ 11, 1, 5, 8, 0, 4, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*247*/ 11, 1, 5, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*248*/ 5, 1, 9, 1, 10, 9, 10, 8, 9, inv, inv, inv, inv, inv, inv, inv,
/*249*/ 10, 4, 1, 9, 5, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*250*/ 8, 1, 10, 0, 1, 8, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*251*/ 10, 4, 1, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*252*/ 9, 4, 8, 5, 4, 9, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*253*/ 9, 5, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*254*/ 4, 8, 0, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
/*255*/ inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv, inv,
);


@compute @workgroup_size(64)
fn computeCellTriangles(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&rawTri)) {
        return;
    }

    let coord = cellCoord(global_id.x);

    let vId = array<u32, 8>(
        coord.z * zVOff + yVOff*coord.y + coord.x,
        coord.z * zVOff + yVOff*coord.y + (coord.x + 1),
        coord.z * zVOff + yVOff*(coord.y+1) + coord.x,
        coord.z * zVOff + yVOff*(coord.y+1) + (coord.x + 1),

        (coord.z+1) * zVOff + yVOff*coord.y + coord.x,
        (coord.z+1) * zVOff + yVOff*coord.y + (coord.x + 1),
        (coord.z+1) * zVOff + yVOff*(coord.y+1) + coord.x,
        (coord.z+1) * zVOff + yVOff*(coord.y+1) + (coord.x+1),
    );

    var caseId = u32(vertices[vId[7]].value >= 0);
    caseId = (caseId << 1) | u32(vertices[vId[6]].value >= 0);
    caseId = (caseId << 1) | u32(vertices[vId[5]].value >= 0);
    caseId = (caseId << 1) | u32(vertices[vId[4]].value >= 0);
    caseId = (caseId << 1) | u32(vertices[vId[3]].value >= 0);
    caseId = (caseId << 1) | u32(vertices[vId[2]].value >= 0);
    caseId = (caseId << 1) | u32(vertices[vId[1]].value >= 0);
    caseId = (caseId << 1) | u32(vertices[vId[0]].value >= 0);

    var off = caseId * 16;
    let triOff = global_id.x * 15;
    for(var i = 0u; i < 15; i++) {
        let id = triTable[off + i];
        if(id != inv) {
            // rawTri[triOff + i] = cellEdge(coord, id);
            rawTri[triOff + i] = edgeChunkOffset[cellEdge(coord, id)];
        } else {
            rawTri[triOff + i] = inv;
        }
    }


    // rawTri[global_id.x * 15] = caseId;
    // rawTri[global_id.x * 15 + 1] = off;
    // rawTri[global_id.x * 15 + 1] = coord.x;
    // rawTri[global_id.x * 15 + 2] = coord.y;
    // rawTri[global_id.x * 15 + 3] = coord.z;
    // rawTri[global_id.x * 15 + 2] = cellEdge(coord, 0);
    // rawTri[global_id.x * 15 + 3] = cellEdge(coord, 1);
    // rawTri[global_id.x * 15 + 4] = cellEdge(coord, 2);
    // rawTri[global_id.x * 15 + 5] = cellEdge(coord, 3);
    // rawTri[global_id.x * 15 + 6] = cellEdge(coord, 4);
    // rawTri[global_id.x * 15 + 7] = cellEdge(coord, 5);
    // rawTri[global_id.x * 15 + 8] = cellEdge(coord, 6);
    // rawTri[global_id.x * 15 + 9] = cellEdge(coord, 7);
    // rawTri[global_id.x * 15 + 10] = cellEdge(coord, 8);
    // rawTri[global_id.x * 15 + 11] = cellEdge(coord, 9);
    // rawTri[global_id.x * 15 + 12] = cellEdge(coord, 10);
    // rawTri[global_id.x * 15 + 13] = cellEdge(coord, 11);

    _ = X;
    _ = Y;
    _ = Z;
    _ = CHUNKSIZE;
}

@compute @workgroup_size(64)
fn countTrianglesPerChunk(@builtin(global_invocation_id) global_id : vec3<u32>) {
    _ = X;
    _ = Y;
    _ = Z;
    _ = CHUNKSIZE;

	if(global_id.x >= arrayLength(&indexCount) - 1) {
        return;
    }

    let chunkOffset = global_id.x * CHUNKSIZE;
    var localOffset = 0u;
    for(var i = 0u; i < CHUNKSIZE; i++) {
		// for(var t = 0u; t < 15; t++) {
			if(rawTri[chunkOffset + i ] != INVALID) {
				rawTriOffset[chunkOffset + i ] = localOffset;
				localOffset++;
			} 
			else {
				// break;
				rawTriOffset[chunkOffset + i ] = INVALID;
			}
		// }
	}

	indexCount[global_id.x] = localOffset;
}

@compute @workgroup_size(1)
fn reduceTriangleCount() {
    var off0 = indexCount[0];
    var off1 = 0u;
    indexCount[0] = 0;
    for(var i = 0u; i < arrayLength(&indexCount) - 1; i++) {
        off1 = indexCount[i+1];
        indexCount[i+1] = indexCount[i] + off0;
        off0 = off1;
    }

    _ = X;
    _ = Y;
    _ = Z;
    _ = CHUNKSIZE;
}

@compute @workgroup_size(64)
fn completeTriangleOffsets(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if(global_id.x >= arrayLength(&rawTriOffset)
        || rawTriOffset[global_id.x] == INVALID) {
        return;
    }

    let chunkId = global_id.x / CHUNKSIZE;
    let chunkOffset = indexCount[chunkId];

    rawTriOffset[global_id.x] += chunkOffset ;

    _ = X;
    _ = Y;
    _ = Z;
    _ = CHUNKSIZE;
}

// @compute @workgroup_size(64)
// fn copyToIndexBuffer(@builtin(global_invocation_id) global_id : vec3<u32>) {


//     _ = X;
//     _ = Y;
//     _ = Z;
//     _ = CHUNKSIZE;
// }