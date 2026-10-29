#!/usr/bin/env python3
"""Render a .glb to a PNG with a tiny software rasterizer (verification only)."""
import json, struct, sys, math
import numpy as np
from PIL import Image

def load(path):
    d=open(path,'rb').read()
    assert d[:4]==b'glTF'
    off=12; js=None; bins=None
    while off < len(d):
        ln,ty=struct.unpack_from('<II',d,off); off+=8
        chunk=d[off:off+ln]; off+=ln
        if ty==0x4E4F534A: js=json.loads(chunk)
        else: bins=chunk
    return js,bins

def acc(js,bins,i):
    a=js['accessors'][i]; bv=js['bufferViews'][a['bufferView']]
    off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    n=a['count']; ct=a['componentType']; ty=a['type']
    comps={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[ty]
    dt={5126:'<f4',5125:'<u4',5123:'<u2'}[ct]
    arr=np.frombuffer(bins,dtype=dt,count=n*comps,offset=off)
    return arr.reshape(n,comps) if comps>1 else arr

def main(path,out,W=900,H=600,views=((35,20),(150,15),(90,80),(0,0))):
    js,bins=load(path)
    tris=[];cols=[]
    for node in js['nodes']:
        if 'mesh' not in node: continue
        off=np.array(node.get('translation',[0,0,0]),dtype=np.float64)
        mesh=js['meshes'][node['mesh']]
        for p in mesh['primitives']:
            P=acc(js,bins,p['attributes']['POSITION']).astype(np.float64)+off
            I=acc(js,bins,p['indices']).reshape(-1,3)
            m=js['materials'][p['material']]
            c=m['pbrMetallicRoughness']['baseColorFactor']
            tris.append(P[I]); cols.append(c)
    T=np.concatenate(tris); C=np.array([c[:3] for c in cols])
    # per-tri material colour
    csum=np.array([len(t) for t in tris])
    tri_col=np.repeat(C,csum,axis=0)
    lo=T.reshape(-1,3).min(0); hi=T.reshape(-1,3).max(0)
    ctr=(lo+hi)/2; rad=float(np.linalg.norm(hi-lo))/2
    imgs=[]
    light=np.array([0.4,0.8,0.5]); light/=np.linalg.norm(light)
    for (az,el) in views:
        img=np.zeros((H,W,3),dtype=np.float64)+np.array([0.08,0.10,0.14])
        zb=np.full((H,W),1e9)
        a=math.radians(az); e=math.radians(el)
        fwd=np.array([math.cos(e)*math.sin(a),math.sin(e),math.cos(e)*math.cos(a)])
        eye=ctr+fwd*rad*3.2
        zc=(eye-ctr); zc/=np.linalg.norm(zc)
        up=np.array([0,1,0.0])
        xc=np.cross(up,zc); xc/=np.linalg.norm(xc)
        yc=np.cross(zc,xc)
        R=np.stack([xc,yc,zc])
        f=1.3
        V=((T.reshape(-1,3)-eye)@R.T).reshape(-1,3,3)
        sx=W/2+ (V[:,:,0]/ -V[:,:,2])*f*H/2
        sy=H/2- (V[:,:,1]/ -V[:,:,2])*f*H/2
        z=-V[:,:,2]
        # normals for shading
        N=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0])
        N/= (np.linalg.norm(N,axis=1,keepdims=True)+1e-12)
        lam=np.clip(N@light,0,1)*0.75+0.28
        shade=tri_col*lam[:,None]
        for i in range(len(T)):
            if z[i].min()<=0.01: continue
            xs=sx[i]; ys=sy[i]
            x0=max(int(xs.min()),0); x1=min(int(xs.max())+1,W-1)
            y0=max(int(ys.min()),0); y1=min(int(ys.max())+1,H-1)
            if x0>x1 or y0>y1: continue
            d=(ys[1]-ys[2])*(xs[0]-xs[2])+(xs[2]-xs[1])*(ys[0]-ys[2])
            if abs(d)<1e-9: continue
            X,Y=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1))
            l0=((ys[1]-ys[2])*(X-xs[2])+(xs[2]-xs[1])*(Y-ys[2]))/d
            l1=((ys[2]-ys[0])*(X-xs[2])+(xs[0]-xs[2])*(Y-ys[2]))/d
            l2=1-l0-l1
            msk=(l0>=0)&(l1>=0)&(l2>=0)
            if not msk.any(): continue
            zz=l0*z[i,0]+l1*z[i,1]+l2*z[i,2]
            sub=zb[y0:y1+1,x0:x1+1]
            upd=msk&(zz<sub)
            sub[upd]=zz[upd]
            im=img[y0:y1+1,x0:x1+1]
            im[upd]=shade[i]
        imgs.append((img*255).clip(0,255).astype(np.uint8))
    sheet=np.concatenate([np.concatenate(imgs[:2],axis=1),np.concatenate(imgs[2:4],axis=1)],axis=0)
    Image.fromarray(sheet).save(out)
    print('wrote',out,[ (round(x,2)) for x in (lo.tolist()+hi.tolist())])

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])
