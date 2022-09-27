#version 330 core

out vec4 out_color;

in vec2 v_texcoord;

const float PI = 3.14159265359f;
const uint NUM_SAMPLES = 1024u;

float saturate(float f)
{
	return clamp(f, 0.0f, 1.0f);
}
float RadicalInverse_VdC(uint bits) // In place of bitfieldreverse()
{
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);

	return float(bits) * 2.3283064365386963e-10;
}
vec2 ComputeHammersley(uint i, uint N)
{
	return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
vec3 ComputeImportanceSampleGGX(vec2 Xi, float roughness, vec3 N)
{
	float alpha = roughness * roughness;

	float phi = 2 * PI * Xi.x;
	float cos_theta = sqrt((1.0f - Xi.y) / (1.0f + (alpha * alpha - 1.0f) * Xi.y));
	float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

	vec3 H;
	H.x = sin_theta * cos(phi);
	H.y = sin_theta * sin(phi);
	H.z = cos_theta;

	vec3 up = abs(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 tan_x = normalize(cross(up, N));
	vec3 tan_y = cross(N, tan_x);

	return normalize(tan_x * H.x + tan_y * H.y + N * H.z);
}
float ComputeGeometryAttenuationGGXSmith(float NdotL, float NdotV, float roughness) // Should use another G function here
{
	float NdotL2 = NdotL * NdotL;
	float NdotV2 = NdotV * NdotV;
	float kRough2 = roughness * roughness + 0.0001f;

	float ggxL = (2.0f * NdotL) / (NdotL + sqrt(NdotL2 + kRough2 * (1.0f - NdotL2)));
	float ggxV = (2.0f * NdotV) / (NdotV + sqrt(NdotV2 + kRough2 * (1.0f - NdotV2)));

	return ggxL * ggxV;
}
vec2 ComputeIntegrateBRDF(float NdotV, float roughness)
{
	vec3 V;
	V.x = sqrt(1.0f - NdotV * NdotV);
	V.y = 0.0f;
	V.z = NdotV;

	vec3 N = vec3(0.0f, 0.0f, 1.0f);
	vec2 brdfLUT = vec2(0.0f);

	for(uint i = 0u; i < NUM_SAMPLES; i++)
	{
		vec2 Xi = ComputeHammersley(i, NUM_SAMPLES);
		vec3 H = ComputeImportanceSampleGGX(Xi, roughness, N);
		vec3 L = normalize(2.0f * dot(V, H) * H - V);

		float NdotL = saturate(L.z);
		float NdotH = saturate(H.z);
		float VdotH = saturate(dot(V, H));

		if(NdotL > 0.0f)
		{
			float G = ComputeGeometryAttenuationGGXSmith(NdotL, NdotV, roughness);
			float G_Vis = (G * VdotH) / (NdotH * NdotV);
			float Fc = pow(1.0f - VdotH, 5.0f);

			brdfLUT.x += (1.0f - Fc) * G_Vis;
			brdfLUT.y += Fc * G_Vis;
		}
	}

	return brdfLUT / NUM_SAMPLES;
}

void main()
{
	vec2 integrated_brdf = ComputeIntegrateBRDF(v_texcoord.x, v_texcoord.y);

	out_color = vec4(integrated_brdf, 1.0, 1.0);
}