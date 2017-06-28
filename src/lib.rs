#[macro_use]
extern crate glium;
extern crate unicode_normalization;
extern crate rusttype;

use std::borrow::Cow;
use std::cell::RefCell;

use glium::texture::Texture2d;
use glium::backend::Facade;
use glium::backend::glutin_backend::GlutinFacade;

pub use rusttype::{Scale, point, Point, vector, Vector, Rect, SharedBytes};
use rusttype::{Font, FontCollection, PositionedGlyph, Glyph};
use rusttype::gpu_cache::*;



pub struct Render<'a> {
    fonts: Vec<Font<'a>>,
    pub texture: Texture2d,
    cache: Cache,
    hidpi_factor: f32,
    tolerance: (f32, f32),
}
impl<'a> Render<'a> {
    pub fn new(gl: &GlutinFacade) -> Self {
        let dpi = gl.get_window().unwrap().hidpi_factor();
        // We can always just resize it if it's too small.
        let initial_size = 512.0 * dpi;
        let tolerance = (0.1, 0.1);

        let mut ret = Render {
            fonts: Vec::new(),
            texture: Texture2d::empty(gl, initial_size as u32, initial_size as u32).unwrap(),
            cache: Cache::new(0, 0, tolerance.0, tolerance.1),
            hidpi_factor: dpi, // FIXME: ask window + may need updating?
            tolerance: tolerance,
        };
        ret.set_cache(gl, initial_size as u32);
        ret
    }

    pub fn add_fonts<B>(&mut self, bytes: B)
    where B: Into<SharedBytes<'a>>,
    {
        let fonts = FontCollection::from_bytes(bytes);
        self.fonts.extend(fonts.into_fonts());
    }

    fn set_cache<G: Facade>(&mut self, gl: &G, size: u32) {
        let dim = (size, size);
        if self.cache.dimensions() != dim {
            self.cache = Cache::new(size, size, self.tolerance.0, self.tolerance.1);
        }
        if self.texture.dimensions() != dim {
            self.texture = Texture2d::empty(gl, size, size).unwrap();
        }
    }
}

// type UiText<'a> = Text<'a, (i32, i32)>;
pub struct Text<'a, P> {
    /// Where this text is located in space. In 2D applications, `(i32, i32)` is quite sufficient.
    /// 3D applications will likely need some point3, orientation, `right`, and `down`.
    pub pos: P,
    pub width: u32,
    pub text: Cow<'a, str>,
    pub style: Style,
}

#[derive(Debug, Clone, Copy)]
pub enum Style {
    Normal,
    // FIXME: This is incomplete.
}
impl Style {
    fn get_scale(self, render: &Render) -> Scale {
        Scale::uniform(16.0 * render.hidpi_factor)
    }
}

pub struct Buffer<'a, P> {
    parts: Vec<Text<'a, P>>,
}
impl<'a, P> Buffer<'a, P>
where P: 'a,
{
    pub fn new() -> Self {
        Buffer {
            parts: Vec::new(),
        }
    }
    pub fn push(&mut self, t: Text<'a, P>) {
        self.parts.push(t);
    }

    pub fn write<S>(&mut self, pos: P, text: S, width: u32)
    where S: Into<Cow<'a, str>>
    {
        self.push(Text {
            pos: pos,
            width: width,
            text: text.into(),
            style: Style::Normal,
        });
    }

    pub fn reset(&mut self) {
        self.parts.clear();
    }

    /// You're in charge of building the vertex buffer.
    pub fn build<'b, F>(&mut self, render: &mut Render, mut f: F)
    where
        F: for<'x, 'y> FnMut(&'x Text<'y, P>, Rect<i32>, Rect<f32>),
    {
        let fonts = &render.fonts;
        let mut glyphs = &mut Vec::new();
        for text in &self.parts {
            let scale = text.style.get_scale(render);
            let fontid = 0; // FIXME: Font selection
            Self::layout_paragraph(
                glyphs,
                fonts,
                scale,
                text.width as i32,
                text.text.as_ref(),
            );
            for glyph in glyphs.iter() {
                render.cache.queue_glyph(fontid, glyph.clone());
            }
            let texture = &mut render.texture;
            render.cache.cache_queued(|rect, data| {
                texture.main_level().write(glium::Rect {
                    left: rect.min.x,
                    bottom: rect.min.y,
                    width: rect.width(),
                    height: rect.height(),
                }, glium::texture::RawImage2d {
                    data: Cow::Borrowed(data),
                    width: rect.width(),
                    height: rect.height(),
                    format: glium::texture::ClientFormat::U8,
                });
            }).unwrap();
            for glyph in glyphs.iter() {
                if let Ok(Some((uv, pos))) = render.cache.rect_for(fontid, glyph) {
                    f(text, pos, uv);
                }
            }

            glyphs.clear();
        }
    }

    fn layout_paragraph<'f, 'r>(
        result: &'r mut Vec<PositionedGlyph<'f>>,
        fonts: &'f Vec<Font<'f>>,
        scale: Scale,
        width: i32,
        text: &str,
    ) -> Point<f32>
    where 'f: 'r
    {
        use self::unicode_normalization::UnicodeNormalization;
        let v_metrics = fonts[0].v_metrics(scale);
        let advance_height = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap;
        let max_area = RefCell::new(point(0.0, 0.0));
        let caret = RefCell::new(point(0.0, v_metrics.ascent));
        let replacement_char = get_glyph('�', fonts).or_else(|| get_glyph('?', fonts));
        let mut unit_start = 0;
        let mut any_break = false;
        let bump_line = || {
            let mut max_area = max_area.borrow_mut();
            let mut caret = caret.borrow_mut();
            if caret.x > max_area.x { max_area.x = caret.x; }
            *caret = point(0.0, caret.y + advance_height);
            max_area.y = caret.y;
        };
        let is_separator = |c: char| {
            c == ' '
        };
        let mut last_glyph_id = None;
        for c in text.nfc() {
            if c.is_control() {
                if c == '\n' {
                    bump_line();
                    unit_start = result.len();
                    any_break = false;
                }
                continue;
            }
            if is_separator(c) {
                unit_start = result.len() + 1; // We break at the *next* character
                any_break = true;
            }
            let (font, base_glyph) = match get_glyph(c, fonts).or_else(|| replacement_char.clone()) {
                Some((font, glyph)) => (font, glyph),
                None => continue,
            };
            if let Some(id) = last_glyph_id.take() {
                caret.borrow_mut().x += font.pair_kerning(scale, id, base_glyph.id());
            }
            last_glyph_id = Some(base_glyph.id());
            let mut glyph: PositionedGlyph = base_glyph.scaled(scale).positioned(*caret.borrow());
            if let Some(bb) = glyph.pixel_bounding_box() {
                if bb.max.x > width {
                    bump_line();
                    if result.len() > unit_start && any_break {
                        // There's probably some weird degenerate case where this'd panic w/o this
                        // check.
                        let mut caret = caret.borrow_mut();
                        let delta = *caret - result[unit_start].position();
                        for g in &mut result[unit_start..] {
                            *g = g.clone().into_unpositioned().positioned(g.position() + delta);
                        }
                        let last = result.last().expect("any glyphs");
                        *caret = last.position();
                        caret.x += last.unpositioned().h_metrics().advance_width;
                        any_break = false;
                    }
                    glyph = glyph.into_unpositioned().positioned(*caret.borrow());
                    last_glyph_id = None;
                }
            }
            caret.borrow_mut().x += glyph.unpositioned().h_metrics().advance_width;
            result.push(glyph);
        }
        let ret = max_area.borrow().clone();
        ret
    }

}


fn get_glyph<'a, 'f>(c: char, fonts: &'f Vec<Font<'f>>) -> Option<(&'f Font<'a>, Glyph<'f>)>
where 'f: 'a
{
    // FIXME: Try fallback fonts.
    let font = &fonts[0];
    font.glyph(c).map(|g| (font, g))
}

pub mod simple2d {
    use super::*;

    use glium;
    use glium::backend::Facade;
    use glium::{Program, Surface};
    use glium::program::ProgramChooserCreationError;

    /// Something to get you up & running.
    pub struct Simple2d {
        program: Program,
    }
    impl Simple2d {
        pub fn create<G: Facade>(gl: &G) -> Result<Simple2d, ProgramChooserCreationError> {
           let program = program!(
                gl,
                // FIXME: The other versions.
                140 => {
                    vertex: "
                        #version 140

                        in vec2 position;
                        in vec2 tex_coords;
                        in vec4 color;

                        out vec2 v_tex_coords;
                        out vec4 v_color;

                        void main() {
                            gl_Position = vec4(position, 0.0, 1.0);
                            v_tex_coords = tex_coords;
                            v_color = color;
                        }
                    ",

                    fragment: "
                        #version 140
                        uniform sampler2D tex;
                        in vec2 v_tex_coords;
                        in vec4 v_color;
                        out vec4 f_color;

                        void main() {
                            f_color = v_color * vec4(1.0, 1.0, 1.0, texture(tex, v_tex_coords).r);
                        }
                    "
                }
           );
           program.map(|p| Simple2d {
               program: p,
           })
        }

        pub fn draw<G: Facade>(
            &self,
            gl: &G,
            target: &mut glium::Frame,
            font: &mut Render,
            buffer: &mut Buffer<(i32, i32)>
        ) -> Result<(), glium::DrawError>
        {
            let (screen_width, screen_height) = target.get_dimensions();
            let (screen_width, screen_height) = (screen_width as f32, screen_height as f32);
            let color = [0.0, 0.0, 0.0, 1.0];
            let mut vertices = Vec::new();
            buffer.build(font, |text, pos_rect, uv_rect| {
                let origin = vector(text.pos.0, text.pos.1);
                let (gl_rect_min, gl_rect_max) = (
                    2.0 * vector(
                        (pos_rect.min.x + origin.x) as f32 / screen_width - 0.5,
                        1.0 - (pos_rect.min.y + origin.y) as f32 / screen_height - 0.5,
                    ),
                    2.0 * vector(
                        (pos_rect.max.x + origin.x) as f32 / screen_width - 0.5,
                        1.0 - (pos_rect.max.y + origin.y) as f32 / screen_height - 0.5,
                    ),
                );
                let verts = [
                    Vertex {
                        position: [gl_rect_min.x, gl_rect_max.y],
                        tex_coords: [uv_rect.min.x, uv_rect.max.y],
                        color: color
                    },
                    Vertex {
                        position: [gl_rect_min.x,  gl_rect_min.y],
                        tex_coords: [uv_rect.min.x, uv_rect.min.y],
                        color: color
                    },
                    Vertex {
                        position: [gl_rect_max.x,  gl_rect_min.y],
                        tex_coords: [uv_rect.max.x, uv_rect.min.y],
                        color: color
                    },
                    Vertex {
                        position: [gl_rect_max.x,  gl_rect_min.y],
                        tex_coords: [uv_rect.max.x, uv_rect.min.y],
                        color: color
                    },
                    Vertex {
                        position: [gl_rect_max.x, gl_rect_max.y],
                        tex_coords: [uv_rect.max.x, uv_rect.max.y],
                        color: color
                    },
                    Vertex {
                        position: [gl_rect_min.x, gl_rect_max.y],
                        tex_coords: [uv_rect.min.x, uv_rect.max.y],
                        color: color
                    },
                ];
                vertices.extend_from_slice(&verts[..]);
            });

            let vbo = glium::VertexBuffer::new(gl, &vertices).unwrap();
            let uniforms = uniform! {
                tex: font.texture.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Nearest)
            };
            target.draw(
                &vbo,
                glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
                &self.program,
                &uniforms,
                &glium::DrawParameters {
                    blend: glium::Blend::alpha_blending(),
                    .. Default::default()
                },
            )
        }
    }


    #[derive(Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
        tex_coords: [f32; 2],
        color: [f32; 4]
    }

    implement_vertex!(Vertex, position, tex_coords, color);
}
