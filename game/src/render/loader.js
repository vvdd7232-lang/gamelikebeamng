// Minimal .glb (binary glTF 2.0) loader: positions / normals / indices / materials.

function parseGlb(buffer) {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a glb file');
  let off = 12;
  let json = null;
  let bin = null;
  while (off < dv.byteLength) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    off += 8;
    const chunk = new Uint8Array(buffer, off, len);
    if (type === 0x4e4f534a) {
      json = JSON.parse(new TextDecoder().decode(chunk));
    } else if (type === 0x004e4942) {
      bin = chunk;
    }
    off += len;
    if (len % 4) off += 4 - (len % 4);
  }
  return { json, bin };
}

function accessor(json, bin, index) {
  const a = json.accessors[index];
  const bv = json.bufferViews[a.bufferView];
  const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }[a.type];
  const base = (bv.byteOffset || 0) + (a.byteOffset || 0);
  const n = a.count * comps;
  let out;
  if (a.componentType === 5126) {
    out = new Float32Array(bin.buffer.slice(bin.byteOffset + base, bin.byteOffset + base + n * 4));
  } else if (a.componentType === 5125) {
    out = new Uint32Array(bin.buffer.slice(bin.byteOffset + base, bin.byteOffset + base + n * 4));
  } else if (a.componentType === 5123) {
    out = new Uint16Array(bin.buffer.slice(bin.byteOffset + base, bin.byteOffset + base + n * 2));
  } else {
    throw new Error('unsupported component type ' + a.componentType);
  }
  return { data: out, comps, count: a.count };
}

/**
 * Load a glb into { nodes: [{name, translation, primitives:[{positions, normals,
 * indices, material}]}], materials: [...] }
 */
export async function loadGlb(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
  const buffer = await res.arrayBuffer();
  const { json, bin } = parseGlb(buffer);
  const materials = (json.materials || []).map((m) => {
    const p = m.pbrMetallicRoughness || {};
    const c = p.baseColorFactor || [0.7, 0.7, 0.7, 1];
    return {
      name: m.name || 'mat',
      color: [c[0], c[1], c[2]],
      alpha: c[3] === undefined ? 1 : c[3],
      metallic: p.metallicFactor === undefined ? 0.2 : p.metallicFactor,
      roughness: p.roughnessFactor === undefined ? 0.6 : p.roughnessFactor,
      emissive: m.emissiveFactor ? [m.emissiveFactor[0], m.emissiveFactor[1], m.emissiveFactor[2]] : [0, 0, 0],
      blend: m.alphaMode === 'BLEND',
    };
  });
  const nodes = json.nodes.map((n) => {
    const prims = [];
    if (n.mesh !== undefined) {
      for (const p of json.meshes[n.mesh].primitives) {
        const pos = accessor(json, bin, p.attributes.POSITION);
        const nor = p.attributes.NORMAL !== undefined ? accessor(json, bin, p.attributes.NORMAL) : null;
        const idx = p.indices !== undefined ? accessor(json, bin, p.indices) : null;
        prims.push({
          positions: pos.data,
          normals: nor ? nor.data : null,
          indices: idx ? idx.data : null,
          material: p.material !== undefined ? materials[p.material] : null,
        });
      }
    }
    return { name: n.name, translation: n.translation || [0, 0, 0], primitives: prims };
  });
  return { nodes, materials };
}
