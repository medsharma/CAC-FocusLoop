/// Metal port of the Axonal Labs website fragment shader (`brain.js`).
/// Kept as source and compiled at runtime so it works no matter how the
/// project's targets/bundles are organised (no `.metal` file or default library needed).
///
/// Differences from the website:
///   * activity comes from 7 Yeo-network weights · activations (or a per-vertex buffer)
///     instead of 4 hand-placed Gaussian blobs;
///   * `gain` (default 0.85) replaces the fixed 1.25 so a fully active network peaks
///     orange-yellow instead of clipping to white.
enum BrainShaderSource {
    static let metal = """
    #include <metal_stdlib>
    using namespace metal;

    struct VIn {
        float3 pos  [[attribute(0)]];
        float3 nrm  [[attribute(1)]];
        float4 w0   [[attribute(2)]];   // networks 1-4
        float4 w1   [[attribute(3)]];   // networks 5-7, .w = baked ambient occlusion
        float  vact [[attribute(4)]];   // optional per-vertex activity (fsaverage5 order)
    };

    struct Uniforms {
        float4x4 proj;
        float4x4 rot;
        float4   act0;     // network activations 1-4
        float4   act1;     // network activations 5-7, .w unused
        float4   params;   // x = time, y = shimmer, z = gain, w = vertex-activity mix
    };

    struct VOut {
        float4 position [[position]];
        float3 n;
        float3 vp;
        float  act;
        float  ao;
    };

    vertex VOut brainVertex(VIn v [[stage_in]], constant Uniforms& u [[buffer(2)]]) {
        VOut o;
        float3 p  = (u.rot * float4(v.pos, 1.0)).xyz;
        float3 vp = p - float3(0.0, 0.0, 4.6);
        o.position = u.proj * float4(vp, 1.0);
        o.n  = (u.rot * float4(v.nrm, 0.0)).xyz;
        o.vp = vp;
        float net = dot(v.w0, u.act0) + dot(float4(v.w1.xyz, 0.0), u.act1);
        o.act = mix(net, v.vact, u.params.w);
        o.ao  = v.w1.w;
        return o;
    }

    fragment float4 brainFragment(VOut f [[stage_in]], constant Uniforms& u [[buffer(0)]]) {
        float3 n = normalize(f.n);
        float v = clamp(f.act * u.params.z, 0.0, 1.2);
        v *= 0.6 + 0.4 * f.ao;                                   // activity rides the gyral crowns
        v += u.params.y * sin(u.params.x * 2.0 + dot(f.vp, float3(5.1, 7.3, 6.2)));
        v = clamp(v, 0.0, 1.0);
        float3 heat = mix(float3(0.3, 0.04, 0.01), float3(0.7, 0.08, 0.01), smoothstep(0.05, 0.35, v));
        heat = mix(heat, float3(0.95, 0.38, 0.04), smoothstep(0.35, 0.62, v));
        heat = mix(heat, float3(1.0, 0.88, 0.5),   smoothstep(0.62, 0.8, v));
        heat = mix(heat, float3(1.0, 1.0, 1.0),    smoothstep(0.78, 0.96, v));
        float3 albedo = mix(float3(0.97, 0.97, 0.98), heat, smoothstep(0.2, 0.42, v));
        albedo *= 0.6 + 0.4 * f.ao;                              // baked sulcal shadows
        float3 l  = normalize(float3(0.45, 0.6, 0.65));
        float3 l2 = normalize(float3(-0.6, 0.15, -0.35));
        float diff = max(dot(n, l), 0.0);
        float fill = max(dot(n, l2), 0.0);
        float spec = pow(max(dot(reflect(-l, n), normalize(-f.vp)), 0.0), 42.0) * 0.13;
        float3 c = albedo * (0.44 + 0.54 * diff + 0.2 * fill) + float3(spec);
        return float4(c, 1.0);
    }
    """
}
