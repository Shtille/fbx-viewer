#version 330 core

uniform vec3 u_to_light;
uniform float u_g;
uniform float u_g2;

out vec4 color;

in vec3 v_mie_color;
in vec3 v_rayleigh_color;
in vec3 v_to_camera;

void main()
{
	float cos_a = dot(u_to_light, v_to_camera) / length(v_to_camera);
	float rayleigh_phase = 1.0 + cos_a * cos_a;
	float mie_phase = (1.0 - u_g2) / (2.0 + u_g2) * (1.0 + cos_a * cos_a) / pow(1.0 + u_g2 - 2.0 * u_g * cos_a, 1.5);
	color = vec4(1.0 - exp(-1.5 * (rayleigh_phase * v_rayleigh_color + mie_phase * v_mie_color)), 1.0);
}

