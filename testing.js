console.log("testing");

// const INTMAX = 123456789;

// const nbEdges = 100;
// const activeEdgeBuffer = new Array(nbEdges);
// for(let i = 0; i < activeEdgeBuffer.length; ++i) {
//     activeEdgeBuffer[i] = Math.random() > 0.75 ? i : null;
// }
// console.log(activeEdgeBuffer)

// console.table(activeEdgeBuffer)

// const chunkSize = 16;
// const nbChunks = Math.ceil(nbEdges / chunkSize);
// const activePerChunkBuffer = new Array(nbChunks);
// console.log(nbChunks, nbChunks * chunkSize)


// const activeEdgeOffsetBuffer = new Array(nbEdges);
// for(let ch = 0; ch < nbChunks; ++ch) {
//     const chunkOffset = ch * chunkSize;
//     let subOffset = 0;
//     for(let i = 0; i < chunkSize; ++i) {
//         const e = chunkOffset + i;
//         if(e >= nbEdges)
//             continue;

//         if(activeEdgeBuffer[e]) {
//             activeEdgeOffsetBuffer[e] = subOffset++;
//         } else {
//             activeEdgeOffsetBuffer[e] = null;
//         }
//     }

//     activePerChunkBuffer[ch] = subOffset;
// }

// console.table([activeEdgeBuffer, activeEdgeOffsetBuffer])
// console.table(activePerChunkBuffer)
// const chunkOffsetBuffer = new Array(nbChunks);
// chunkOffsetBuffer[0] = 0;
// for(let ch = 1; ch < nbChunks + 1; ++ch) {
//     chunkOffsetBuffer[ch] = chunkOffsetBuffer[ch - 1] + activePerChunkBuffer[ch - 1];
// }

// console.table([activePerChunkBuffer, chunkOffsetBuffer])


// const vertexBuffer = new Array((nbEdges))
// console.log(vertexBuffer.length)

// for(let i = 0; i < activeEdgeBuffer.length; ++i) {
//     const ch = Math.floor(i / chunkSize);
//     const chunkOffset = chunkOffsetBuffer[ch];
//     if(activeEdgeBuffer[i]) {
//         const edgeOffset = activeEdgeOffsetBuffer[i];
//         const vertex = chunkOffset + edgeOffset;
//         console.log(vertex)
//         activeEdgeBuffer[i] = vertex
//         vertexBuffer[vertex] = (vertexBuffer[vertex] ?? 0) +1;
//     }
// }
// console.table(vertexBuffer);
// console.table([activeEdgeBuffer, activeEdgeOffsetBuffer])


const X = 2, Y = 2, Z = 2;
const nbEdges = (X+1)*(Y+1)*Z +(X+1)*(Z+1)*Y +(Z+1)*(Y+1)*X; 
// console.log(nbEdges)
const yEOff = X*(Y+1)*(Z+1);
const zEOff = yEOff+(X+1)*Y*(Z+1);
const xEOff =  X*(Y+1);
const xEOff1 =  (X+1)*(Y+1);
function cellEdges(x, y, z) {
    const e0 = x + X * y + xEOff * z;
    const e1 = x + X *(y+1) + xEOff * z;
    const e2 = x + X*y + xEOff*(z+1);
    const e3 = x + X *(y+1) + xEOff*(z+1);
    // console.log(e0, e1, e2, e3);

    const e4 = yEOff + x + (X+1)*y+ xEOff * z;
    const e5 = yEOff + (x+1) + (X+1)*y+ xEOff * z;
    const e6 = yEOff + x + (X+1)*y+ xEOff * (z+1);
    const e7 = yEOff + (x+1) + (X+1)*y+ xEOff * (z+1);
    // console.log(e4, e5, e6, e7);

    const e8 = zEOff + x + (X+1)*y + xEOff1 * z;
    const e9 = zEOff + (x+1) + (X+1)*y + xEOff1 * z;
    const e10 = zEOff + x + (X+1)*(y+1) + xEOff1 * z;
    const e11 = zEOff + (x+1) + (X+1)*(y+1) + xEOff1 * z;
    // console.log(e8, e9, e10, e11);

    // console.log(X*(Y+1)*(Z+1))
    // console.log((X+1)*Y*(Z+1))
}

cellEdges(0, 1, 1)
const yVOff = (X + 1)
const zVOff = yVOff * (Y + 1);
function cellVertices(x, y, z) {
    const v0 = z * zVOff + yVOff*y + x;
    const v1 = z * zVOff + yVOff*y + (x + 1);
    const v2 = z * zVOff + yVOff*(y+1) + x;
    const v3 = z * zVOff + yVOff*(y+1) + (x + 1);

    const v4 = (z+1) * zVOff + yVOff*y + x;
    const v5 = (z+1) * zVOff + yVOff*y + (x + 1);
    const v6 = (z+1) * zVOff + yVOff*(y+1) + x;
    const v7 = (z+1) * zVOff + yVOff*(y+1) + (x+1);

    // console.log(v0, v1, v2, v3, v4, v5, v6, v7);
}

cellVertices(1, 1, 1)

const nbX = (X)*(Y+1)*(Z+1);
const nbY = (X+1)*(Y)*(Z+1);
const nbZ = (X+1)*(Y+1)*(Z);
function edgeCoord(e) {
    // console.log(`edge ${e}`);
    const A = (e >= nbX) * nbX + (e >= (nbX + nbY)) * nbY;
    const B = (X + (e > nbX)) * (Y + ((e < nbX) || (e > nbX + nbY)));
    const C = X + (e > nbX)


    const Ex = 1*(e < nbX);
    const Ey = 1*(e < nbY + nbX && e >= nbX);
    const Ez = 1*(e >= nbX + nbY);

    let e_ = e - A;
    let z = Math.floor(e_ / B)
    e_ = e_ % B;
    let y = Math.floor(e_ / C);
    let x = e_ % C;
    return {x, y, z};
}

// console.log(edgeCoord(49));
// console.log(edgeCoord(30));
// console.log(edgeCoord(25));
// console.log(edgeCoord(15));
// console.log(edgeCoord(18));
// console.log(edgeCoord(36));
// console.log(edgeCoord(17));
// console.log(edgeCoord(35));

function vertex(x, y, z){
let v = x + (X+1) * y + (X+1)*(Y+1) * z
// console.log(v)
vertexCoord(v)
return v;
}


function vertexCoord(v) {
// console.log('vertex', v)
let z = Math.floor(v / ((X+1)*(Y+1)));
v %= (X+1)*(Y+1);
let y = Math.floor(v / (X+1))
let x = v % (X+1);
// console.log(x, y, z)
}

// vertexCoord(26)

function edgeVertices(e) {
    let eCoord0 = edgeCoord(e);
    let eCoord1 = edgeCoord(e);

    if(e < nbX)
        eCoord1.x += 1;
    else if(e < nbY + nbX)
        eCoord1.y += 1;
    else
        eCoord1.z += 1;

    console.log(vertex(eCoord0.x, eCoord0.y, eCoord0.z), vertex(eCoord1.x, eCoord1.y, eCoord1.z))
}

edgeVertices(36)
edgeVertices(35)
edgeVertices(0)
edgeVertices(17)
edgeVertices(30)
edgeVertices(53)