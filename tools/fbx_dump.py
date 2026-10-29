import struct, zlib, sys, collections

class Node:
    __slots__=('name','props','children')
    def __init__(self,name,props,children):
        self.name=name; self.props=props; self.children=children
    def find(self,n):
        return [c for c in self.children if c.name==n]

def read_props(d,off,n):
    props=[]
    for _ in range(n):
        t=chr(d[off]); off+=1
        if t=='Y': v=struct.unpack_from('<h',d,off)[0]; off+=2
        elif t=='C': v=bool(d[off]); off+=1
        elif t=='I': v=struct.unpack_from('<i',d,off)[0]; off+=4
        elif t=='F': v=struct.unpack_from('<f',d,off)[0]; off+=4
        elif t=='D': v=struct.unpack_from('<d',d,off)[0]; off+=8
        elif t=='L': v=struct.unpack_from('<q',d,off)[0]; off+=8
        elif t in 'SR':
            l=struct.unpack_from('<I',d,off)[0]; off+=4
            raw=d[off:off+l]; off+=l
            v=raw if t=='R' else raw.decode('utf8','replace')
        elif t in 'fdlib':
            al,enc,cl=struct.unpack_from('<III',d,off); off+=12
            raw=d[off:off+cl]; off+=cl
            if enc==1: raw=zlib.decompress(raw)
            fmt={'f':'f','d':'d','l':'q','i':'i','b':'b'}[t]
            v=list(struct.unpack_from('<%d%s'%(al,fmt),raw,0))
        else:
            raise ValueError('bad prop type %r at %d'%(t,off))
        props.append(v)
    return props,off

def read_node(d,off):
    if off+13>len(d): return None,off
    end,np_,pl=struct.unpack_from('<III',d,off)
    nl=d[off+12]
    if end==0 and np_==0 and pl==0 and nl==0: return None, off+13
    name=d[off+13:off+13+nl].decode('utf8','replace')
    off+=13+nl
    props,off=read_props(d,off,np_)
    children=[]
    while off<end:
        c,off2=read_node(d,off)
        if c is None: off=off2; break
        children.append(c); off=off2
    return Node(name,props,children), end

def parse(path):
    d=open(path,'rb').read()
    off=27
    nodes=[]
    while off<len(d):
        n,off2=read_node(d,off)
        if n is None: break
        nodes.append(n); off=off2
    return nodes

if __name__=='__main__':
    ns=parse(sys.argv[1])
    def walk(n,depth=0,maxd=int(sys.argv[2]) if len(sys.argv)>2 else 3):
        print('  '*depth+n.name+' ['+', '.join((p if isinstance(p,str) else str(type(p).__name__)+'('+str(len(p))+')' if isinstance(p,list) else str(p)) [:60] for p in n.props)+']')
        if depth<maxd:
            for c in n.children: walk(c,depth+1,maxd)
    for n in ns: walk(n)
