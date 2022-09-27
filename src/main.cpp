#include "ufbx.h"

#include "model/mesh.h"
#include "graphics/text.h"
#include "camera.h"

#include "declare_main.h"

#include <cmath>

#include <cstdio>

int main()
{
	// Load FBX scene
	ufbx_load_opts opts = { 0 }; // Optional, pass NULL for defaults
	ufbx_error error; // Optional, pass NULL if you don't care about errors
	ufbx_scene *scene = ufbx_load_file("data/scenes/cube_animation_test.fbx", &opts, &error);
	if (!scene) {
		fprintf(stderr, "Failed to load: %s\n", error.description.data);
		return 1;
	}

	// Don't forget to free the allocation!
	ufbx_free_scene(scene);
	return 0;
}

/*
The main concept of creating this application is testing cubemap edge artefacts during accessing mipmaps.
*/
namespace {
	const int kShadowMapSize = 1024;
}

class FbxViewerApp
: public scythe::OpenGlApplication
, public scythe::DesktopInputListener
{
public:
	FbxViewerApp()
	: quad_(nullptr)
	, sphere_(nullptr)
	, cube_(nullptr)
	, font_(nullptr)
	, fps_text_(nullptr)
	, camera_manager_(nullptr)
	, angle_(0.0f)
	, light_distance_(10.0f)
	, need_update_projection_matrix_(true)
	, show_shadow_texture_(false)
	, is_vsm_(true)
	{
		SetInputListener(this);
	}
	const char* GetTitle() final
	{
		return "FBX Viewer";
	}
	const bool IsMultisample() final
	{
		return true;
	}
	void BindShaderConstants()
	{
		const scythe::Vector3 kLightColor(1.0f);

		quad_shader_->Bind();
		quad_shader_->Uniform1i("u_texture", 0);

		if (is_vsm_)
		{
			blur_shader_->Bind();
			blur_shader_->Uniform1i("u_texture", 0);
		}

		object_shader_->Bind();
		object_shader_->Uniform3fv("u_light.color", kLightColor);
		object_shader_->Uniform1i("u_shadow_sampler", 0);
		object_shader_->Unbind();
	}
	void BindShaderVariables()
	{
		object_shader_->Bind();
		object_shader_->Uniform3fv("u_light.position", light_position_);
		object_shader_->Unbind();
	}
	bool Load() final
	{
		// Vertex formats
		scythe::VertexFormat * quad_vertex_format;
		{
			scythe::VertexAttribute attributes[] = {
				scythe::VertexAttribute(scythe::VertexAttribute::kVertex, 3)
			};
			renderer_->AddVertexFormat(quad_vertex_format, attributes, _countof(attributes));
		}
		scythe::VertexFormat * object_vertex_format;
		{
			scythe::VertexAttribute attributes[] = {
				scythe::VertexAttribute(scythe::VertexAttribute::kVertex, 3),
				scythe::VertexAttribute(scythe::VertexAttribute::kNormal, 3)
			};
			renderer_->AddVertexFormat(object_vertex_format, attributes, _countof(attributes));
		}

		// Quad model
		quad_ = new scythe::Mesh(renderer_);
		quad_->CreateQuadFullscreen();
		if (!quad_->MakeRenderable(quad_vertex_format))
			return false;

		// Sphere model
		sphere_ = new scythe::Mesh(renderer_);
		sphere_->CreateSphere(1.0f, 128, 64);
		if (!sphere_->MakeRenderable(object_vertex_format))
			return false;

		// Cube model
		cube_ = new scythe::Mesh(renderer_);
		cube_->CreateCube();
		if (!cube_->MakeRenderable(object_vertex_format))
			return false;
		
		// Load shaders
		if (!renderer_->AddShader(text_shader_, "data/shaders/text")) return false;
		if (!renderer_->AddShader(quad_shader_, "data/shaders/quad")) return false;
		if (is_vsm_)
		{
			if (!renderer_->AddShader(object_shader_, "data/shaders/shadows/object_vsm")) return false;
			if (!renderer_->AddShader(object_shadow_shader_, "data/shaders/shadows/depth_vsm")) return false;
			if (!renderer_->AddShader(blur_shader_, "data/shaders/blur")) return false;
		}
		else
		{
			if (!renderer_->AddShader(object_shader_, "data/shaders/shadows/object")) return false;
			if (!renderer_->AddShader(object_shadow_shader_, "data/shaders/shadows/object_shadow")) return false;
		}

		// Render targets
		if (is_vsm_)
		{
			renderer_->AddRenderTarget(shadow_color_rt_, kShadowMapSize, kShadowMapSize, scythe::Image::Format::kRG32);
			renderer_->AddRenderDepthStencil(shadow_depth_rt_, kShadowMapSize, kShadowMapSize, 32, 0);
			renderer_->AddRenderTarget(blur_color_rt_, kShadowMapSize, kShadowMapSize, scythe::Image::Format::kRG32);
		}
		else
			renderer_->CreateTextureDepth(shadow_depth_rt_, kShadowMapSize, kShadowMapSize, 32);

		// Fonts
		renderer_->AddFont(font_, "data/fonts/GoodDog.otf");
		if (font_ == nullptr)
			return false;

		fps_text_ = scythe::DynamicText::Create(renderer_, 30);
		if (!fps_text_)
			return false;

		camera_manager_ = new scythe::CameraManager();
		camera_manager_->MakeFree(scythe::Vector3(5.0f), scythe::Vector3(0.0f));

		// Finally bind constants
		BindShaderConstants();

		// Load FBX scene
		ufbx_load_opts opts = { 0 }; // Optional, pass NULL for defaults
		ufbx_error error; // Optional, pass NULL if you don't care about errors
		ufbx_scene *scene = ufbx_load_file("data/scenes/cube_animation_test.fbx", &opts, &error);
		if (!scene) {
			fprintf(stderr, "Failed to load: %s\n", error.description.data);
			return false;
		}

		// Don't forget to free the allocation!
		ufbx_free_scene(scene);
		
		return true;
	}
	void Unload() final
	{
		if (camera_manager_)
			delete camera_manager_;
		if (fps_text_)
			delete fps_text_;
		if (quad_)
			delete quad_;
		if (sphere_)
			delete sphere_;
		if (cube_)
			delete cube_;
	}
	void Update() final
	{
		const float kFrameTime = GetFrameTime();

		camera_manager_->Update(kFrameTime);

		angle_ += 0.1f * kFrameTime;
		light_position_.Set(light_distance_ * cosf(angle_), 1.0f, light_distance_ * sinf(angle_));
		//light_position_.Set(1.0f, light_distance_, 1.0f);

		// Update matrices
		renderer_->SetViewMatrix(camera_manager_->view_matrix());
		UpdateProjectionMatrix();
		projection_view_matrix_ = renderer_->projection_matrix() * renderer_->view_matrix();

		BindShaderVariables();
	}
	void RenderObjects(scythe::Shader * shader)
	{
		// Render cube
		renderer_->PushMatrix();
		shader->UniformMatrix4fv("u_model", renderer_->model_matrix());
		cube_->Render();
		renderer_->PopMatrix();

		renderer_->PushMatrix();
		renderer_->Translate(0.0f, -6.0f, 0.0f);
		renderer_->Scale(5.0f);
		shader->UniformMatrix4fv("u_model", renderer_->model_matrix());
		cube_->Render();
		renderer_->PopMatrix();

		// Render spheres
		renderer_->PushMatrix();
		renderer_->Translate(2.0f, 0.0f, 0.0f);
		shader->UniformMatrix4fv("u_model", renderer_->model_matrix());
		sphere_->Render();
		renderer_->PopMatrix();

		renderer_->PushMatrix();
		renderer_->Translate(-2.0f, 0.0f, 0.0f);
		shader->UniformMatrix4fv("u_model", renderer_->model_matrix());
		sphere_->Render();
		renderer_->PopMatrix();

		renderer_->PushMatrix();
		renderer_->Translate(0.0f, 0.0f, 2.0f);
		shader->UniformMatrix4fv("u_model", renderer_->model_matrix());
		sphere_->Render();
		renderer_->PopMatrix();

		renderer_->PushMatrix();
		renderer_->Translate(0.0f, 0.0f, -2.0f);
		shader->UniformMatrix4fv("u_model", renderer_->model_matrix());
		sphere_->Render();
		renderer_->PopMatrix();
	}
	void ShadowPass()
	{
		// Generate matrix for shadows
		// Ortho matrix is used for directional light sources and perspective for spot ones.
		float znear = 1.0f;
		float zfar = light_distance_ * 2.0f;
		scythe::Matrix4 depth_projection;
		scythe::Matrix4::CreatePerspective(45.0f, 1.0f, znear, zfar, &depth_projection);
		scythe::Matrix4 depth_view;
		scythe::Matrix4::CreateLookAt(light_position_, scythe::Vector3(0.0f), scythe::Vector3::UnitY(), &depth_view);
		scythe::Matrix4 depth_projection_view = depth_projection * depth_view;
		/*
			Native view of bias matrix is:
			    | 0.5 0.0 0.0 0.5 |
			M = | 0.0 0.5 0.0 0.5 |
			    | 0.0 0.0 0.5 0.5 |
			    | 0.0 0.0 0.0 1.0 |
		*/
		scythe::Matrix4 bias_matrix(
			0.5f, 0.0f, 0.0f, 0.5f,
			0.0f, 0.5f, 0.0f, 0.5f,
			0.0f, 0.0f, 0.5f, 0.5f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
		depth_bias_projection_view_matrix_ = bias_matrix * depth_projection_view;

		if (is_vsm_)
			renderer_->ChangeRenderTarget(shadow_color_rt_, shadow_depth_rt_);
		else
			renderer_->ChangeRenderTarget(nullptr, shadow_depth_rt_);
		renderer_->ClearColorAndDepthBuffers();

		object_shadow_shader_->Bind();
		object_shadow_shader_->UniformMatrix4fv("u_projection_view", depth_projection_view);

		RenderObjects(object_shadow_shader_);

		object_shadow_shader_->Unbind();

		renderer_->ChangeRenderTarget(nullptr, nullptr);

		if (is_vsm_)
		{
			const float kBlurScale = 1.0f;
			const float kBlurSize = kBlurScale / static_cast<float>(kShadowMapSize);

			renderer_->DisableDepthTest();
			blur_shader_->Bind();

			// Blur horizontally
			renderer_->ChangeRenderTarget(blur_color_rt_, nullptr);
			renderer_->ChangeTexture(shadow_color_rt_, 0);
			renderer_->ClearColorBuffer();
			blur_shader_->Uniform2f("u_scale", kBlurSize, 0.0f);
			quad_->Render();

			// Blur vertically
			renderer_->ChangeRenderTarget(shadow_color_rt_, nullptr);
			renderer_->ChangeTexture(blur_color_rt_, 0);
			renderer_->ClearColorBuffer();
			blur_shader_->Uniform2f("u_scale", 0.0f, kBlurSize);
			quad_->Render();

			// Back to main framebuffer
			renderer_->ChangeRenderTarget(nullptr, nullptr);

			blur_shader_->Unbind();
			renderer_->EnableDepthTest();
		}
	}
	void RenderScene()
	{
		// First render shadows
		if (is_vsm_)
		{
			ShadowPass();
		}
		else
		{
			renderer_->CullFace(scythe::CullFaceType::kFront);
			ShadowPass();
			renderer_->CullFace(scythe::CullFaceType::kBack);
		}

		if (show_shadow_texture_)
		{
			quad_shader_->Bind();

			if (is_vsm_)
				renderer_->ChangeTexture(shadow_color_rt_, 0);
			else
				renderer_->ChangeTexture(shadow_depth_rt_, 0);
			quad_->Render();
			renderer_->ChangeTexture(nullptr, 0);

			quad_shader_->Unbind();
		}
		else
		{
			// Render objects
			object_shader_->Bind();
			object_shader_->UniformMatrix4fv("u_projection_view", projection_view_matrix_);
			object_shader_->UniformMatrix4fv("u_depth_bias_projection_view", depth_bias_projection_view_matrix_);

			if (is_vsm_)
				renderer_->ChangeTexture(shadow_color_rt_, 0);
			else
				renderer_->ChangeTexture(shadow_depth_rt_, 0);

			RenderObjects(object_shader_);

			renderer_->ChangeTexture(nullptr, 0);

			object_shader_->Unbind();
		}
	}
	void RenderInterface()
	{
		renderer_->DisableDepthTest();
		
		// Draw FPS
		text_shader_->Bind();
		text_shader_->Uniform1i("u_texture", 0);
		text_shader_->Uniform4f("u_color", 1.0f, 0.5f, 1.0f, 1.0f);
		fps_text_->SetText(font_, 0.0f, 0.8f, 0.05f, L"fps: %.2f", GetFrameRate());
		fps_text_->Render();
		text_shader_->Unbind();

		renderer_->ChangeTexture(nullptr);
		
		renderer_->EnableDepthTest();
	}
	void Render() final
	{
		renderer_->SetViewport(width_, height_);

		renderer_->ClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		renderer_->ClearColorAndDepthBuffers();

		RenderScene();
		RenderInterface();
	}
	void OnChar(unsigned short code) final
	{
	}
	void OnKeyDown(scythe::PublicKey key, int mods) final
	{
		if (key == scythe::PublicKey::kF)
		{
			DesktopApplication::ToggleFullscreen();
		}
		else if (key == scythe::PublicKey::kEscape)
		{
			DesktopApplication::Terminate();
		}
		else if (key == scythe::PublicKey::kF5)
		{
			renderer_->TakeScreenshot("screenshots");
		}
		else if (key == scythe::PublicKey::kLeft)
		{
			camera_manager_->RotateAroundTargetInY(0.1f);
		}
		else if (key == scythe::PublicKey::kRight)
		{
			camera_manager_->RotateAroundTargetInY(-0.1f);
		}
		else if (key == scythe::PublicKey::kUp)
		{
			camera_manager_->RotateAroundTargetInZ(0.1f);
		}
		else if (key == scythe::PublicKey::kDown)
		{
			camera_manager_->RotateAroundTargetInZ(-0.1f);
		}
		else if (key == scythe::PublicKey::kSpace)
		{
			show_shadow_texture_ = !show_shadow_texture_;
		}
	}
	void OnKeyUp(scythe::PublicKey key, int modifiers) final
	{
	}
	void OnMouseDown(scythe::MouseButton button, int modifiers) final
	{
	}
	void OnMouseUp(scythe::MouseButton button, int modifiers) final
	{
	}
	void OnMouseMove() final
	{
	}
	void OnScroll(float delta_x, float delta_y) final
	{
	}
	void OnSize(int w, int h) final
	{
		DesktopApplication::OnSize(w, h);
		// To have correct perspective when resizing
		need_update_projection_matrix_ = true;
	}
	void UpdateProjectionMatrix()
	{
		if (need_update_projection_matrix_ || camera_manager_->animated())
		{
			need_update_projection_matrix_ = false;
			scythe::Matrix4 projection_matrix;
			scythe::Matrix4::CreatePerspective(45.0f, aspect_ratio_, 0.1f, 100.0f, &projection_matrix);
			renderer_->SetProjectionMatrix(projection_matrix);
		}
	}
	
private:
	scythe::Mesh * quad_;
	scythe::Mesh * sphere_;
	scythe::Mesh * cube_;

	scythe::Shader * text_shader_;
	scythe::Shader * quad_shader_;
	scythe::Shader * object_shader_;
	scythe::Shader * object_shadow_shader_; //!< simplified version for shadows generation
	scythe::Shader * blur_shader_;

	scythe::Texture * shadow_color_rt_;
	scythe::Texture * shadow_depth_rt_;
	scythe::Texture * blur_color_rt_;

	scythe::Font * font_;
	scythe::DynamicText * fps_text_;
	scythe::CameraManager * camera_manager_;
	
	scythe::Matrix4 projection_view_matrix_;
	scythe::Matrix4 depth_bias_projection_view_matrix_;
	scythe::Vector3 light_position_;

	float angle_;
	const float light_distance_;

	bool need_update_projection_matrix_;
	bool show_shadow_texture_;
	const bool is_vsm_;
};

//DECLARE_MAIN(FbxViewerApp);