#version 330 core

uniform sampler2D u_texture;

out vec4 out_color;

in vec2 v_texcoord;

void main()
{
	out_color = texture(u_texture, v_texcoord);
}