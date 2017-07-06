#[macro_use]
extern crate glium;
extern crate unicode_normalization;
extern crate rusttype;

use std::borrow::Cow;
use std::ops::Range;

use glium::texture::Texture2d;
use glium::backend::Facade;
use glium::backend::glutin_backend::GlutinFacade;

pub use rusttype::{Scale, point, Point, vector, Vector, Rect, SharedBytes};
use rusttype::{Font, FontCollection, PositionedGlyph, Glyph, GlyphId};
use rusttype::gpu_cache::*;



pub struct Render<'a> {
    fonts: Vec<Font<'a>>,
    pub texture: Texture2d,
    cache: Cache,
    hidpi_factor: f32,
    tolerance: (f32, f32),
}
impl<'a> Render<'a> {
    /// texture_size: 512, tolerance: (0.1, 0.1)
    pub fn new(gl: &GlutinFacade, texture_size: u32, tolerance: (f32, f32)) -> Self {
        let dpi = gl.get_window().unwrap().hidpi_factor();
        // We can always just resize it if it's too small.
        let initial_size = (texture_size as f32 * dpi) as u32;

        let mut ret = Render {
            fonts: Vec::new(),
            texture: Texture2d::empty_with_format(
                gl,
                glium::texture::UncompressedFloatFormat::U8,
                glium::texture::MipmapsOption::NoMipmap,
                initial_size,
                initial_size,
            ).unwrap(),
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

pub struct Text<'a> {
    pub style: Style,
    pub text: Cow<'a, str>,
}
impl<'a, I> From<(Style, I)> for Text<'a>
where
    I: Into<Cow<'a, str>>,
{
    fn from((s, t): (Style, I)) -> Self {
        Text {
            style: s,
            text: t.into(),
        }
    }
}
// Can't do From<Cow>...
impl<'a> From<String> for Text<'a> {
    fn from(t: String) -> Self {
        Text {
            style: Style::default(),
            text: t.into(),
        }
    }
}
impl<'a> From<&'a str> for Text<'a> {
    fn from(t: &'a str) -> Self {
        Text {
            style: Style::default(),
            text: t.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Style {
    /// The size of the font. The default scale is 16.0.
    pub scale: f32,
    /// RGBA color components, from 0 to 255.
    pub color: (u8, u8, u8, u8),
    /// NYI
    pub fontid: usize,

    /// Mark the text for drawing with shadows. This is not implemented here.
    pub shadow: bool,
    /// NYI
    pub bold: bool,
    /// NYI
    pub italic: bool,
    /// NYI
    pub underline: bool,
    /// NYI
    pub strike: bool,
}
impl Default for Style {
    fn default() -> Self {
        Style {
            scale: 24.0,
            color: (0, 0, 0, 0xFF),
            fontid: 0,

            shadow: false,
            bold: false,
            italic: false,
            underline: false,
            strike: false,
        }
    }
}
impl Style {
    pub fn new() -> Self {
        Self::default()
    }

    fn get_glyph<'context, 'fonts>(&self, c: char, fonts: &'context Vec<Font<'fonts>>) -> Option<(usize, Glyph<'fonts>)>
        where 'context: 'fonts
    {
        // FIXME: Try fallback fonts.
        // FIXME: How do we figure out who the bold fonts are?
        let font = &fonts[self.fontid];
        font.glyph(c).map(|g| (self.fontid, g))
    }
}

pub enum FlowControl {
    Next,
    Skip,
    Stop,
}

pub struct Layout<P> {
    pub origin: P,
    pub width: u32,
}
impl<P> Layout<P> {
    fn to_block(&self) -> LayoutBlock {
        LayoutBlock {
            width: self.width,
            caret: point(0.0, 0.0),
            unit_start: 0,
            any_break: false,
            max_area: point(0.0, 0.0),
            last_glyph_id: None,
        }
    }
}

struct LayoutBlock {
    width: u32,
    caret: Point<f32>,
    unit_start: usize,
    any_break: bool,
    max_area: Point<f32>,
    last_glyph_id: Option<GlyphId>,
}
impl LayoutBlock {
    fn bump_line(&mut self, advance_height: f32) {
        if self.caret.x > self.max_area.x { self.max_area.x = self.caret.x; }
        self.caret = point(0.0, self.caret.y + advance_height);
        self.max_area.y = self.caret.y;
    }
}


pub struct RunBuffer<'text, P> {
    parts: Vec<Text<'text>>,
    runs: Vec<(Range<usize>, Layout<P>)>,
    char_count: usize,
}
impl<'text, P> RunBuffer<'text, P>
where
    P: 'text,
{
    pub fn new() -> Self {
        RunBuffer {
            parts: Vec::new(),
            runs: Vec::new(),
            char_count: 0,
        }
    }

    pub fn push_run<I>(&mut self, layout: Layout<P>, it: I)
    where I: Iterator<Item=Text<'text>>
    {
        let start = self.parts.len();
        for text in it {
            self.char_count += text.text.len();
            self.parts.push(text);
        }
        let end = self.parts.len();
        self.runs.push((start..end, layout));
    }

    pub fn push(&mut self, layout: Layout<P>, t: Text<'text>) {
        self.push_run(layout, Some(t).into_iter())
    }

    pub fn write<S>(&mut self, pos: P, width: u32, text: S)
    where S: Into<Cow<'text, str>>
    {
        self.push(
            Layout {
                origin: pos,
                width: width
            },
            Text {
                text: text.into(),
                style: Style::default(),
            },
        );
    }

    pub fn reset(&mut self) {
        self.parts.clear();
        self.runs.clear();
        self.char_count = 0;
    }

    /// Returns an estimate of how many glyphs are in the buffer. This may be inaccurate due to eg
    /// ligatures and normalization.
    pub fn glyph_estimate(&self) -> usize {
        self.char_count
    }

    /// Runs the layout algorithm on the most recently added run, and returns the amount of space
    /// used, and its `Layout` object.
    pub fn measure_area<'fonts>(&mut self, render: &'fonts mut Render) -> (Point<f32>, &mut Layout<P>) {
        let &mut(ref range, ref mut layout) = self.runs.last_mut().expect("measure_height called with any run");
        let mut layout_block = layout.to_block();
        let mut glyphs = Vec::new();
        let mut max_area = point(0.0, 0.0);
        for text in &self.parts[range.clone()] {
            let nm = layout_paragraph(
                &mut layout_block,
                &mut glyphs,
                render.hidpi_factor,
                &render.fonts,
                text,
            );
            if nm.x > max_area.x { max_area.x = nm.x; }
            if nm.y > max_area.y { max_area.y = nm.y; }
        }
        (max_area, layout)
    }

    /// Runs the layout algorithm, and uploads glyphs to the cache.
    /// 
    /// `write_glyph` is called with each positioned glyph. You can use this to build your vertex
    /// buffer.
    /// 
    /// The first parameter holds the `Text` and `Style`.
    /// The second parameter is the `origin`, `&P`.
    /// The third parameter specifies the rectangle the glyph occupises, relative to the `origin`.
    /// For example, if P is a point in a 2D UI, then the glyph is simply positioned at `origin + offset`.
    /// The fourth parameter is UV coordinates.
    pub fn build<'fonts, F>(
        &mut self,
        render: &'fonts mut Render,
        mut write_glyph: F
    )
    where
        F: for<'x, 'y> FnMut(&'x Text<'y>, &P, Rect<i32>, Rect<f32>) -> FlowControl,
    {
        let mut glyphs = Vec::new();
        let mut glyph_run: Vec<(&Layout<P>, &Text, Range<usize>)> = Vec::new();
        for &(ref range, ref layout) in &self.runs {
            let mut layout_block = layout.to_block();
            for text in &self.parts[range.clone()] {
                let start = glyphs.len();
                layout_paragraph(
                    &mut layout_block,
                    &mut glyphs,
                    render.hidpi_factor,
                    &render.fonts,
                    text,
                );
                let end = glyphs.len();
                glyph_run.push((layout, text, start..end));
            }
        }
        for &(fontid, ref glyph) in &glyphs {
            render.cache.queue_glyph(fontid, glyph.clone());
        }
        let texture = &mut render.texture;
        // I think this is O(total # of glyphs) rather than O(total # of unique glyphs), but the
        // two numbers were actually fairly similar. (There are extra variants due to sub-pixel
        // positioning?)
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
        for (layout, text, range) in glyph_run {
            for &(fontid, ref glyph) in &glyphs[range] {
                if let Ok(Some((uv, pos))) = render.cache.rect_for(fontid, glyph) {
                    match write_glyph(text, &layout.origin, pos, uv) {
                        FlowControl::Next => (),
                        FlowControl::Skip => break,
                        FlowControl::Stop => return,
                    }
                }
            }
        }
    }
}
fn layout_paragraph<'layout, 'context, 'fonts, 'result, 'text>(
    layout: &'layout mut LayoutBlock,
    result: &'result mut Vec<(usize, PositionedGlyph<'fonts>)>,
    dpi: f32,
    fonts: &'context Vec<Font<'fonts>>,
    text: &'text Text<'text>,
) -> Point<f32>
where
    'context: 'fonts,
{
    let style = &text.style;
    let scale = Scale::uniform(text.style.scale * dpi);
    let v_metrics = fonts[0].v_metrics(scale);
    let advance_height = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap; // FIXME: max(this line)?
    let replacement_char = style.get_glyph('�', fonts).or_else(|| style.get_glyph('?', fonts));
    let is_separator = |c: char| {
        c == ' '
    };
    use self::unicode_normalization::UnicodeNormalization;
    for c in text.text.nfc() {
        if c.is_control() {
            if c == '\n' {
                layout.bump_line(advance_height);
                layout.unit_start = result.len();
                layout.any_break = false;
            }
            continue;
        }
        if is_separator(c) {
            layout.unit_start = result.len() + 1; // We break at the *next* character
            layout.any_break = true;
        }
        let (fontid, base_glyph) = match style.get_glyph(c, fonts).or_else(|| replacement_char.clone()) {
            Some((fontid, glyph)) => (fontid, glyph),
            None => continue,
        };
        let font = &fonts[fontid];
        if let Some(id) = layout.last_glyph_id.take() {
            layout.caret.x += font.pair_kerning(scale, id, base_glyph.id());
        }
        layout.last_glyph_id = Some(base_glyph.id());
        let mut glyph: PositionedGlyph = base_glyph.scaled(scale).positioned(layout.caret);
        if let Some(bb) = glyph.pixel_bounding_box() {
            if bb.max.x as i64 > layout.width as i64 {
                layout.bump_line(advance_height);
                if result.len() > layout.unit_start && layout.any_break {
                    // There's probably some weird degenerate case where this'd panic w/o this
                    // check.
                    let delta = layout.caret - result[layout.unit_start].1.position();
                    for &mut (_, ref mut g) in &mut result[layout.unit_start..] {
                        *g = g.clone().into_unpositioned().positioned(g.position() + delta);
                    }
                    let ref last = result.last().expect("any glyphs").1;
                    layout.caret = last.position();
                    layout.caret.x += last.unpositioned().h_metrics().advance_width;
                    layout.any_break = false;
                }
                glyph = glyph.into_unpositioned().positioned(layout.caret);
                layout.last_glyph_id = None;
            }
        }
        layout.caret.x += glyph.unpositioned().h_metrics().advance_width;
        result.push((fontid, glyph));
    }
    layout.max_area.clone()
}


pub mod simple2d {
    use super::*;

    use glium;
    use glium::backend::Facade;
    use glium::{Program, Surface};
    use glium::program::ProgramChooserCreationError;

    /// Something to get you up & running.
    pub struct Simple2d {
        pub program: Program,
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
            buffer: &mut RunBuffer<(i32, i32)>
        ) -> Result<(), glium::DrawError>
        {
            let (screen_width, screen_height) = target.get_dimensions();
            let (screen_width, screen_height) = (screen_width as f32, screen_height as f32);
            let mut vertices = Vec::new();
            buffer.build(font, |text, origin, pos_rect, uv_rect| {
                let color = {
                    let c = text.style.color;
                    let f = |c| c as f32 / 255.0;
                    [f(c.0), f(c.1), f(c.2), f(c.3)]
                };
                let origin = vector(origin.0, origin.1);
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
                FlowControl::Next
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
