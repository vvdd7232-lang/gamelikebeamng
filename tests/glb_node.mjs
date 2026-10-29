// node-side glb parsing without fetch (mirrors render/loader.js)
export function parseGlbForNode(buffer) {
  const dv = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a glb');
  let off = 12, json = null, bin = null;
  while (off < dv.byteLength) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    off += 8;
    const chunk = new Uint8Array(buffer.buffer, buffer.byteOffset + off, len);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(chunk));
    else if (type === 0x004e4942) bin = chunk;
    off += len;
  }
  const accessor = (index) => {
    const a = json.accessors[index];
    const bv = json.bufferViews[a.bufferView];
    const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[a.type];
    const base = (bv.byteOffset || 0) + (a.byteOffset || 0);
    const n = a.count * comps;
    let out;
    if (a.componentType === 5126) out = new Float32Array(bin.buffer.slice(bin.byteOffset + base, bin.byteOffset + base + n * 4));
    else if (a.componentType === 5125) out = new Uint32Array(bin.buffer.slice(bin.byteOffset + base, bin.byteOffset + base + n * 4));
    else out = new Uint16Array(bin.buffer.slice(bin.byteOffset + base, bin.byteOffset + base + n * 2));
    return { data: out };
  };
  const materials = (json.materials || []).map((m) => {
    const p = m.pbrMetallicRoughness || {};
    const c = p.baseColorFactor || [0.7, 0.7, 0.7, 1];
    return {
      name: m.name || 'mat', color: [c[0], c[1], c[2]],
      alpha: c[3] === undefined ? 1 : c[3],
      metallic: p.metallicFactor === undefined ? 0.2 : p.metallicFactor,
      roughness: p.roughnessFactor === undefined ? 0.6 : p.roughnessFactor,
      emissive: m.emissiveFactor ? m.emissiveFactor.slice(0, 3) : [0, 0, 0],
    };
  });
  const nodes = json.nodes.map((n) => {
    const prims = [];
    if (n.mesh !== undefined) {
      for (const p of json.meshes[n.mesh].primitives) {
        prims.push({
          positions: accessor(p.attributes.POSITION).data,
          normals: p.attributes.NORMAL !== undefined ? accessor(p.attributes.NORMAL).data : null,
          indices: p.indices !== undefined ? accessor(p.indices).data : null,
          material: p.material !== undefined ? materials[p.material] : null,
        });
      }
    }
    return { name: n.name, translation: n.translation || [0, 0, 0], primitives: prims };
  });
  return { nodes, materials };
}
