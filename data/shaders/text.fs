#version 330 core

uniform sampler2D u_texture;
uniform vec4 u_color;

out vec4 color;

in vec2 v_texcoord;

void main()
{
	float alpha = texture(u_texture, v_texcoord).x;
    color = vec4(1.0, 1.0, 1.0, alpha) * u_color;
}