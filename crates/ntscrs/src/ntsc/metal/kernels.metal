#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
struct Params { uint width,height,a,b,c,d,args[8]; float values[16]; };
#define COMMON device float* arena [[buffer(0)]], constant Params& p [[buffer(1)]], device const float* meta [[buffer(2)]]
inline float srgb(float v) { return v<=0.0031308f?v*12.92f:1.055f*pow(v,1.0f/2.4f)-0.055f; }
inline float linear(float v) { return v<=0.04045f?v/12.92f:pow((v+0.055f)/1.055f,2.4f); }
inline float read_component(device const uchar* data,uint component,uint bytes) {
    if(bytes==4) return ((device const float*)data)[component];
    if(bytes==2) return float(((device const ushort*)data)[component])*(1.0f/65535.0f);
    return float(data[component])*(1.0f/255.0f);
}
inline void write_component(device uchar* data,uint component,uint bytes,float value) {
    if(bytes==4) ((device float*)data)[component]=value;
    else if(bytes==2) ((device ushort*)data)[component]=ushort(clamp(value*65535.0f,0.0f,65535.0f));
    else data[component]=uchar(clamp(value*255.0f,0.0f,255.0f));
}
kernel void load_rgb(COMMON,device const uchar* image [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    uint y=id/p.width,x=id%p.width;if(y>=p.height)return;
    uint row=p.args[5]==2?y:y*2+p.args[5];
    if(row>=p.args[0]){arena[p.a+id]=0.0f;arena[p.b+id]=0.0f;arena[p.c+id]=0.0f;return;}
    if(p.args[4])row=p.args[0]-1-row;
    auto pixel=image+row*p.args[1]+x*p.args[2]*p.args[3];
    float r=read_component(pixel,0,p.args[3]),g=read_component(pixel,1,p.args[3]),b=read_component(pixel,2,p.args[3]);
    if(p.args[7]){r=srgb(r);g=srgb(g);b=srgb(b);}
    arena[p.a+id]=fma(r,0.299f,fma(g,0.587f,b*0.114f));
    arena[p.b+id]=fma(r,0.5959f,fma(g,-0.2746f,b*-0.3213f));
    arena[p.c+id]=fma(r,0.2115f,fma(g,-0.5227f,b*0.3112f));
}
kernel void store_rgb(COMMON,device uchar* image [[buffer(3)]],uint id [[thread_position_in_grid]]) {
    uint y=id/p.width,x=id%p.width;
    if(y>=(p.args[6]?p.args[0]:p.height))return;
    uint row=p.args[5]==2||p.args[6]?y:y*2+p.args[5];
    uint r=p.args[6]?min(y/2,p.height-1):y;
    bool interpolate=p.args[6] && (y&1)!=(p.args[5]&1) && y>0 && y+1<p.args[0];
    if(interpolate)r=(y-1)/2;
    uint index=r*p.width+x;
    float3 yiq=float3(arena[p.a+index],arena[p.b+index],arena[p.c+index]);
    if(interpolate) {
        uint other=((y+1)/2)*p.width+x;
        yiq=(yiq+float3(arena[p.a+other],arena[p.b+other],arena[p.c+other]))*0.5f;
    }
    float3 rgb=fma(yiq.z,float3(0.619f,-0.647f,1.703f),fma(yiq.y,float3(0.956f,-0.272f,-1.106f),yiq.x));
    if(p.args[7]){rgb=float3(linear(rgb.r),linear(rgb.g),linear(rgb.b));}
    if(p.args[4])row=p.args[0]-1-row;
    auto pixel=image+row*p.args[1]+x*p.args[2]*p.args[3];
    for(uint c=0;c<3;c++)write_component(pixel,c,p.args[3],rgb[c]);
    if(p.args[2]==4)write_component(pixel,3,p.args[3],1.0f);
}
template<uint Order> inline float filter_sample(float sample,thread float3& z,constant Params& p) {
    float value=fma(p.values[0],sample,z.x);
    z.x=fma(-p.values[4],value,fma(p.values[1],sample,Order>1?z.y:0.0f));
    if(Order>1)z.y=fma(-p.values[5],value,fma(p.values[2],sample,Order>2?z.z:0.0f));
    if(Order>2)z.z=fma(-p.values[6],value,p.values[3]*sample);
    return value;
}
inline float3 initial_state(float value,constant Params& p) {
    if(!p.args[1] || value==0.0f)return 0.0f;
    uint len=p.args[2];float bsum=0.0f,dsum=0.0f;
    for(uint i=1;i<len;i++){bsum+=p.values[i]-p.values[i+3]*p.values[0];dsum+=p.values[i+3];}
    float first=bsum/(dsum+1.0f),asum=1.0f,csum=0.0f;float3 z=0.0f;
    for(uint i=1;i<len-1;i++){asum+=p.values[i+3];csum+=p.values[i]-p.values[i+3]*p.values[0];z[i]=(asum*first-csum)*value;}
    z[0]=first*value;return z;
}
template<uint Order> inline void scan(device float* arena,constant Params& p,device const float* meta,uint row,uint lane) {
    if(row>=p.height)return;
    uint begin=lane*p.args[3],end=min(begin+p.args[3],p.width+p.args[0]);
    auto src=arena+p.a+row*p.width;auto dst=arena+p.b+row*p.width;
    float3 initial=initial_state(src[0],p),state=lane==0?initial:float3(0.0f);
    for(uint x=begin;x<end;x++)filter_sample<Order>(src[min(x,p.width-1)],state,p);
    for(uint power=0;power<5;power++){
        uint offset=1u<<power;float3 left=simd_shuffle_up(state,offset);
        if(lane>=offset){float3 next;
            for(uint i=0;i<3;i++){uint b=power*9+i*3;next[i]=fma(meta[b],left.x,fma(meta[b+1],left.y,fma(meta[b+2],left.z,state[i])));}
            state=next;
        }
    }
    state=simd_shuffle_up(state,1);if(lane==0)state=initial;
    for(uint x=begin;x<end;x++){float value=filter_sample<Order>(src[min(x,p.width-1)],state,p);if(x>=p.args[0])dst[x-p.args[0]]=value;}
}
#define SCAN_KERNEL(name,order) kernel void name(COMMON,uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {scan<order>(arena,p,meta,group*4+simd,lane);}
SCAN_KERNEL(filter_scan,3)
SCAN_KERNEL(filter_scan_1,1)
SCAN_KERNEL(filter_scan_2,2)
constant float imult[4]={1,0,-1,0};constant float qmult[4]={0,1,0,-1};
kernel void modulate(COMMON,uint id [[thread_position_in_grid]]) {
    if(id>=p.width*p.height)return;uint phase=(id%p.width+uint(meta[id/p.width]))&3;
    arena[p.a+id]+=arena[p.b+id]*imult[phase]+arena[p.c+id]*qmult[phase];
}
kernel void demod_luma(COMMON,uint id [[thread_position_in_grid]]) {
    if(id>=p.width*p.height)return;uint y=id/p.width,x=id%p.width;auto src=arena+p.a+y*p.width;
    float value;
    if(p.args[0]==0)value=(((x?src[x-1]:16.0f/255.0f)+src[x])+src[min(x+1,p.width-1)]+src[min(x+2,p.width-1)])*0.25f;
    else if(p.args[0]==1)value=(arena[p.a+(y?y-1:1)*p.width+x]+src[x])*0.5f;
    else if(p.args[0]==2)value=(src[x]*0.5f)+(arena[p.a+(y?y-1:1)*p.width+x]*0.25f)+(arena[p.a+(y+1==p.height?y-1:y+1)*p.width+x]*0.25f);
    else {
        uint prev=y?y-1:1,next=y+1==p.height?y-1:y+1;
        uint left=x<2?x+2:x-2,right=x>p.width-3?x-2:x+2;
        auto top=arena+p.a+prev*p.width;auto bottom=arena+p.a+next*p.width;
        float tl=top[left],t=top[x],tr=top[right],l=src[left],c=src[x],r=src[right],bl=bottom[left],b=bottom[x],br=bottom[right];
        if(p.args[0]==3) value=c*0.75f+((l+r)+(t+b))*0.125f-((tl+tr)+(bl+br))*0.0625f;
        else {
            float gx=(tr-tl)+(r-l)+(br-bl),gy=(bl-tl)+(b-t)+(br-tr);
            float gxt=gx*(1.0f/0.001f),gyt=gy*(1.0f/0.001f);
            float gd1=(gx+gy)*(1.0f/(0.001f*M_SQRT2_F)),gd2=(gx-gy)*(1.0f/(0.001f*M_SQRT2_F));
            float ax=1.0f/(1.0f+gxt*gxt),ay=1.0f/(1.0f+gyt*gyt),norm=0.25f/(ax+ay);
            float wx=ax*norm,wy=ay*norm;
            float wd1=norm/max(1.0f+gd1*gd1,2.0f/(0.001f*0.001f)),wd2=norm/max(1.0f+gd2*gd2,2.0f/(0.001f*0.001f));
            value=c*(0.5f+2.0f*(wd1+wd2))+((l+r)*wx+(t+b)*wy)-((tl+br)*wd1+(tr+bl)*wd2);
        }
    }
    arena[p.b+id]=value;
}
kernel void demod_chroma(COMMON,uint id [[thread_position_in_grid]]) {
    if(id>=p.width*p.height)return;uint x=id%p.width,xi=uint(meta[id/p.width]),phase=(x+xi)&3;
    float center=arena[p.a+id]-arena[p.d+id];float i=center*-imult[phase],q=center*-qmult[phase];
    // Match CPU SIMD's FMA order, including its scalar boundary/remainder path.
    uint simdEnd=1+((p.width>4?(p.width-2)/4:0)*4);
    if(x>=1 && x<simdEnd){
        float l=arena[p.a+id-1]-arena[p.d+id-1],r=arena[p.a+id+1]-arena[p.d+id+1];
        i=fma(l,-imult[(phase+3)&3]*0.5f,i);i=fma(r,-imult[(phase+1)&3]*0.5f,i);
        q=fma(l,-qmult[(phase+3)&3]*0.5f,q);q=fma(r,-qmult[(phase+1)&3]*0.5f,q);
    }else{
        if(x+1<p.width){float r=arena[p.a+id+1]-arena[p.d+id+1];i+=r*-imult[(phase+1)&3]*0.5f;q+=r*-qmult[(phase+1)&3]*0.5f;}
        if(x){float l=arena[p.a+id-1]-arena[p.d+id-1];i+=l*-imult[(phase+3)&3]*0.5f;q+=l*-qmult[(phase+3)&3]*0.5f;}
    }
    arena[p.b+id]=i;arena[p.c+id]=q;
}
inline uint noise_hash(uint v){v=(v^2747636419u)*2654435769u;v=(v^(v>>16))*2654435769u;return(v^(v>>16))*2654435769u;}
inline float gradient(uint hash){uint h=hash>>28;float v=float((h&7)+1);return(h&8)==0?v:-v;}
inline float simplex(float point,int seed){
    float cell=floor(point);int i=int(cell);float x0=point-cell,x1=x0-1.0f;
    float t0=fma(-x0,x0,1.0f),t1=fma(-x1,x1,1.0f);float t20=t0*t0,t21=t1*t1;
    return fma(t20*t20*gradient(noise_hash(uint(i^seed))),x0,t21*t21*gradient(noise_hash(uint((i+1)^seed)))*x1);
}
kernel void noise(COMMON,uint id [[thread_position_in_grid]]) {
    if(id>=p.width*p.height)return;uint row=id/p.width,x=id%p.width;auto m=meta+row*4;
    if(m[3]==0.0f)return;
    // Match the CPU's repeated +4 exactly, including rounding at every exponent
    // boundary. Between boundaries, jump in one exact addition.
    float point=m[1]+float(x&3);uint steps=x>>2;
    while(steps){
        uint exponent=(as_type<uint>(point)>>23)&255u;
        float boundary=as_type<float>((exponent+1u)<<23);
        uint jump=min(steps,max(1u,uint(ceil((boundary-point)*0.25f))));
        point+=float(jump)*4.0f;steps-=jump;
    }
    point*=m[2];int seed=as_type<int>(m[0]);float value=simplex(point,seed);
    for(uint octave=1;octave<p.args[0];octave++){point*=2.0f;value=fma(simplex(point,seed),1.0f,value);}
    arena[p.a+id]+=value*m[3];
}
kernel void rotate_chroma(COMMON,uint id [[thread_position_in_grid]]) {
    if(id>=p.width*p.height)return;uint row=id/p.width;float s=meta[row*2],c=meta[row*2+1];
    float i=arena[p.b+id],q=arena[p.c+id];arena[p.b+id]=i*c-q*s;arena[p.c+id]=i*s+q*c;
}
kernel void shift_plane(COMMON,uint id [[thread_position_in_grid]]) {
    if(id>=p.width*p.height)return;uint y=id/p.width,x=id%p.width;int sy=int(y)-as_type<int>(p.args[1]);
    if(sy<0 || sy>=int(p.height)){arena[p.b+id]=0;return;}
    auto m=meta+y*4;auto src=arena+p.a+uint(sy)*p.width;
    float shift=m[0];int whole=int(shift)-(shift<0.0f?1:0);float frac=shift-float(whole);
    int l=int(x)-whole-1,r=l+1;float boundary=p.args[0]?src[whole>=0?0:p.width-1]:0.0f;
    float left=l>=0&&l<int(p.width)?src[l]:boundary,right=r>=0&&r<int(p.width)?src[r]:boundary;
    int start=max(whole+1,0),upper=int(p.width)-4+min(whole,0),blocks=upper>=start?(upper-start)/4+1:0;
    float value=int(x)>=start&&int(x)<start+blocks*4?fma(left-right,frac,right):left*frac+right*(1.0f-frac);
    if(x<uint(m[1]))value=src[x];
    if(m[2]>0.0f && x>=uint(m[1]) && x<uint(m[1])+uint(ceil(m[2]))){float t=1.0f-float(x-uint(m[1]))/m[2];value+=(t*t)*t*m[3];}
    arena[p.b+id]=value;
}
kernel void add_sparse(COMMON,uint id [[thread_position_in_grid]]) {if(id<p.args[0])arena[p.a+as_type<uint>(meta[id*2])]+=meta[id*2+1];}
kernel void vertical_blend(COMMON,uint id [[thread_position_in_grid]]) {if(id<p.width*p.height)arena[p.b+id]=((id>=p.width?arena[p.a+id-p.width]:0.0f)+arena[p.a+id])*0.5f;}
kernel void drop_chroma(COMMON,uint id [[thread_position_in_grid]]) {if(id<p.width*p.height && meta[id/p.width]!=0.0f){arena[p.b+id]=0.0f;arena[p.c+id]=0.0f;}}
kernel void box_luma(COMMON,uint row [[thread_position_in_grid]]) {
    if(row>=p.height)return;auto src=arena+p.a+row*p.width;auto dst=arena+p.b+row*p.width;
    if(p.width<2){dst[0]=src[0];return;}
    float ring[4]={16.0f/255.0f,16.0f/255.0f,src[0],src[1]};float sum=((ring[0]+ring[1])+ring[2])+ring[3];
    for(uint x=0;x<p.width;x++){float value=src[min(x+2,p.width-1)];sum-=ring[x&3];ring[x&3]=value;sum+=value;dst[x]=sum*0.25f;}
}

kernel void filter_serial(COMMON,uint row [[thread_position_in_grid]]) {
    if(row>=p.height)return;
    auto src=arena+p.a+row*p.width;auto dst=arena+p.b+row*p.width;
    float3 state=initial_state(src[0],p);
    for(uint x=0;x<p.width+p.args[0];x++) {
        float value=filter_sample<3>(src[min(x,p.width-1)],state,p);
        if(x>=p.args[0])dst[x-p.args[0]]=value;
    }
}
