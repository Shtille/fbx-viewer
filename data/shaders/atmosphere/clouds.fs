#version 330 core

uniform sampler2D u_clouds_texture;

out vec4 color;

in vec3 v_color;
in vec3 v_attenuate;
in vec2 v_texcoord;

void main()
{
	float cloudiness = texture(u_clouds_texture, v_texcoord).r;
	color = vec4(v_color + vec3(cloudiness) * v_attenuate, cloudiness);
}