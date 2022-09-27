#version 330 core

uniform samplerCube u_texture;

out vec4 out_color;

smooth in vec3 v_position;

const float PI = 3.14159265359;

void main()
{
	vec3 N = normalize(v_position);
	vec3 up = abs(N.z) < 0.999f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f);
	vec3 right = cross(up, N);
	up = cross(N, right);

	vec3 irradiance_accumulation = vec3(0.0f);

	const float sample_offset = 0.025f;
	float sample_count = 0.0f;

	for (float phi = 0.0f; phi < 2.0f * PI; phi += sample_offset)
	{
		for (float theta = 0.0f; theta < 0.5f * PI; theta += sample_offset)
		{
			vec3 sample_tangent = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
			vec3 sample_vector = sample_tangent.x * right + sample_tangent.y * up + sample_tangent.z * N;

			irradiance_accumulation += texture(u_texture, sample_vector).rgb * cos(theta) * sin(theta);
			sample_count++;
		}
	}

	irradiance_accumulation = irradiance_accumulation * (1.0f / float(sample_count)) * PI;

	out_color = vec4(irradiance_accumulation, 1.0f);
}