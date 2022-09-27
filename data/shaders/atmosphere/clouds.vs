#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;

uniform mat4 u_projection_view;
uniform mat4 u_model;

uniform vec3 u_camera_pos;			// The camera's current position
uniform vec3 u_to_light;			// The direction vector to the light source
uniform vec3 u_inv_wave_length;		// 1 / pow(wavelength, 4) for the red, green, and blue channels
uniform float u_camera_height;		// The camera's current height
uniform float u_camera_height2;		// u_camera_height^2
uniform float u_inner_radius;		// The inner (planetary) radius
uniform float u_outer_radius;		// The outer (atmosphere) radius
uniform float u_outer_radius2;		// u_outer_radius^2
uniform float u_kr_esun;			// Kr * ESun
uniform float u_km_esun;			// Km * ESun
uniform float u_kr_4_pi;			// Kr * 4 * PI
uniform float u_km_4_pi;			// Km * 4 * PI
uniform float u_scale;				// 1 / (u_outer_radius - u_inner_radius)
uniform float u_scale_depth;		// The scale depth (i.e. the altitude at which the atmosphere's average density is found)
uniform float u_scale_over_scale_depth;	// u_scale / u_scale_depth

uniform int u_samples;

uniform bool u_from_space;

out vec3 v_color;
out vec3 v_attenuate;
out vec2 v_texcoord;

float scale(float fCos)
{
	float x = 1.0 - fCos;
	return u_scale_depth * exp(-0.00287 + x * (0.459 + x * (3.83 + x * (-6.80 + x * 5.25))));
}

void main(void)
{
	vec4 world_pos = u_model * vec4(a_position, 1.0);

	// Get the ray from the camera to the vertex, and its length (which is the far point of the ray passing through the atmosphere)
	vec3 pos = world_pos.xyz;
	vec3 ray = pos - u_camera_pos;
	float far = length(ray);
	ray /= far;

	vec3 start;
	float depth;
	if (u_from_space) // from space
	{
		// Calculate the closest intersection of the ray with the outer atmosphere (which is the near point of the ray passing through the atmosphere)
		float B = 2.0 * dot(u_camera_pos, ray);
		float C = u_camera_height2 - u_outer_radius2;
		float det = max(0.0, B * B - 4.0 * C);
		float near = 0.5 * (-B - sqrt(det));

		start = u_camera_pos + ray * near;
		far -= near;
		depth = exp((u_inner_radius - u_outer_radius) / u_scale_depth);
	}
	else // from atmosphere
	{
		start = u_camera_pos;
		depth = exp((u_inner_radius - u_camera_height) / u_scale_depth);
	}

	// Calculate the ray's starting position, then calculate its scattering offset
	// pos/length(pos) is equal to a_normal
	float camera_angle = dot(-ray, pos) / length(pos);
	float light_angle = dot(u_to_light, pos) / length(pos);
	float camera_scale = scale(camera_angle);
	float light_scale = scale(light_angle);
	float camera_offset = depth * camera_scale;
	float sum_scales = light_scale + camera_scale;

	// Initialize the scattering loop variables
	float sample_length = far / u_samples;
	float scaled_length = sample_length * u_scale;
	vec3 sample_ray = ray * sample_length;
	vec3 sample_point = start + sample_ray * 0.5;

	// Now loop through the sample rays
	vec3 front_color = vec3(0.0);
	vec3 attenuate; // attenuation factor for the ground
	for(int i = 0; i < u_samples; i++)
	{
		float height = length(sample_point);
		float depth = exp(u_scale_over_scale_depth * (u_inner_radius - height));
		float scatter = depth * sum_scales - camera_offset;
		attenuate = exp(-scatter * (u_inv_wave_length * u_kr_4_pi + u_km_4_pi));
		front_color += attenuate * (depth * scaled_length);
		sample_point += sample_ray;
	}

	v_color = front_color * (u_inv_wave_length * u_kr_esun + u_km_esun);
	v_attenuate = attenuate;
	v_texcoord = a_texcoord;

	gl_Position = u_projection_view * world_pos;
}
