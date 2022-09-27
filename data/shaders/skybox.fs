#version 330 core

uniform samplerCube u_texture;

out vec4 out_color;

smooth in vec3 v_position;

void main()
{
	out_color = texture(u_texture, v_position);
}