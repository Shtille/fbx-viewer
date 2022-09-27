#version 330 core

uniform sampler2D u_texture;
uniform vec2 u_scale; // blur amount. Should be 1/texture_width

out vec4 out_color;

in vec2 v_texcoord;

void main()
{
	vec4 color = vec4(0.0);
	color += texture(u_texture, v_texcoord + vec2(-3.0*u_scale.x, -3.0*u_scale.y))*0.015625;
	color += texture(u_texture, v_texcoord + vec2(-2.0*u_scale.x, -2.0*u_scale.y))*0.09375;
	color += texture(u_texture, v_texcoord + vec2(-1.0*u_scale.x, -1.0*u_scale.y))*0.234375;
	color += texture(u_texture, v_texcoord + vec2( 0.0          ,  0.0          ))*0.3125;
	color += texture(u_texture, v_texcoord + vec2( 1.0*u_scale.x,  1.0*u_scale.y))*0.234375;
	color += texture(u_texture, v_texcoord + vec2( 2.0*u_scale.x,  2.0*u_scale.y))*0.09375;
	color += texture(u_texture, v_texcoord + vec2( 3.0*u_scale.x,  3.0*u_scale.y))*0.015625;
	out_color = vec4(color.xyz, 1.0);
}