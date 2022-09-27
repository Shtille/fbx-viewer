#version 330 core

uniform sampler2D u_earth_texture;

out vec4 color;

in vec3 v_color;
in vec3 v_attenuate;
in vec2 v_texcoord;

void main()
{
	vec3 earth = texture(u_earth_texture, v_texcoord).rgb;
	color = vec4(v_color + earth * v_attenuate, 1.0);
}