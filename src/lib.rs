#![allow(clippy::nonminimal_bool)]

#[macro_use]
extern crate glium;
extern crate unicode_normalization;
extern crate rusttype;

use std::borrow::Cow;
use std::ops::Range;

use glium::texture::Texture2d;
use glium::backend::Facade;
use glium::backend::glutin::Display;

pub use rusttype::{Scale, point, Point, vector, Vector, Rect, SharedBytes};
use rusttype::{Font, FontCollection, PositionedGlyph, Glyph, GlyphId};
use rusttype::gpu_cache::*;

use self::unicode_normalization::UnicodeNormalization;



pub struct Render<'a> {
    fonts: Vec<Font<'a>>,
    pub texture: Texture2d,
    cache: Cache<'a>,
    hidpi_factor: f64,
    tolerance: (f32, f32),
}
impl<'a> Render<'a> {
    /// texture_size: 512, tolerance: (0.1, 0.1)
    pub fn new(gl: &Display, texture_size: u32, tolerance: (f32, f32)) -> Self {
        let dpi = gl.gl_window().window().scale_factor();
        // We can always just resize it if it's too small.
        let initial_size = (texture_size as f64 * dpi) as u32;

        let mut ret = Render {
            fonts: Vec::new(),
            texture: Texture2d::empty_with_format(
                gl,
                glium::texture::UncompressedFloatFormat::U8,
                glium::texture::MipmapsOption::NoMipmap,
                initial_size,
                initial_size,
            ).unwrap(),
            cache: Cache::builder().build(),
            hidpi_factor: dpi, // FIXME: ask window + may need updating?
            tolerance,
        };
        ret.set_cache(gl, initial_size);
        ret
    }

    pub fn add_fonts<B>(&mut self, bytes: B) -> Result<(), rusttype::Error>
    where
        B: Into<SharedBytes<'a>>,
    {
        let fonts = FontCollection::from_bytes(bytes)?;
        for font in fonts.into_fonts() {
            self.fonts.push(font?);
        }
        Ok(())
    }

    fn set_cache<G: Facade>(&mut self, gl: &G, size: u32) {
        let dim = (size, size);
        if self.cache.dimensions() != dim {
            self.cache = Cache::builder()
                .dimensions(size, size)
                .scale_tolerance(self.tolerance.0)
                .position_tolerance(self.tolerance.1)
                .build();
        }
        if self.texture.dimensions() != dim {
            self.texture = Texture2d::empty(gl, size, size).unwrap();
        }
    }
}

/// A styled `str`.
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
// We can't just impl for S: Into<Cow> CUZ this conflicts with From<(Style, I)>
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
    /// The size of the font. The default scale is 24.0.
    pub scale: f32,
    /// RGBA color components, from 0 to 255.
    pub color: (u8, u8, u8, u8),
    /// NYI
    pub fontid: usize,

    /// Mark the text for drawing with shadows. This is not implemented here.
    pub shadow: Option<(u8, u8, u8, u8)>,
    // FIXME: bitflags
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

            shadow: None,
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

    fn get_glyph<'context, 'fonts>(
        &self,
        c: char,
        fonts: &'context [Font<'fonts>],
    ) -> (usize, Glyph<'fonts>)
    where
        'fonts: 'context,
    {
        // FIXME: Try fallback fonts.
        // FIXME: How do we figure out who the bold fonts are?
        let font = &fonts[self.fontid];
        let g = font.glyph(c);
        (self.fontid, g)
    }
}

pub enum FlowControl {
    NextGlyph,
    SkipBlock,
    StopBuild,
}

/// The bounds of text.
#[derive(Debug, Clone)]
pub struct TextBlock<P> {
    /// The upper-left corner.
    pub origin: P,
    pub width: u32,
}
impl<P> TextBlock<P> {
    fn to_block(&self) -> LayoutBlock {
        LayoutBlock {
            width: self.width,
            caret: point(0.0, 0.0),
            wrap_unit_start: 0,
            any_break: false,
            max_area: point(0.0, 0.0),
            last_glyph_id: None,
            query_area: Rect {
                min: point(0.0, 0.0),
                max: point(0.0, 0.0),
            },
            query_hit: false,
            just_wrapped: false,
        }
    }
}

#[derive(Debug)]
struct LayoutBlock {
    width: u32,
    caret: Point<f32>,
    wrap_unit_start: usize,
    any_break: bool,
    max_area: Point<f32>,
    last_glyph_id: Option<GlyphId>,
    query_area: Rect<f32>,
    query_hit: bool,
    just_wrapped: bool,
}
impl LayoutBlock {
    fn bump_line(&mut self, advance_height: f32) {
        if self.caret.x > self.max_area.x { self.max_area.x = self.caret.x; }
        self.caret = point(0.0, self.caret.y + advance_height);
        self.max_area.y = self.caret.y;
    }
}

#[derive(Default)]
pub struct RunBuffer<'text, P> {
    parts: Vec<Text<'text>>,
    blocks: Vec<(Range<usize>, TextBlock<P>)>,
    char_count: usize,
}
impl<'text, P> RunBuffer<'text, P>
where
    P: 'text,
{
    pub fn new() -> Self {
        RunBuffer {
            parts: Vec::new(),
            blocks: Vec::new(),
            char_count: 0,
        }
    }

    pub fn push_blocks<I>(&mut self, layout: TextBlock<P>, it: I)
    where
        I: Iterator<Item=Text<'text>>,
    {
        let start = self.parts.len();
        for text in it {
            self.char_count += text.text.len();
            self.parts.push(text);
        }
        let end = self.parts.len();
        self.blocks.push((start..end, layout));
    }

    pub fn push<T>(&mut self, layout: TextBlock<P>, t: T)
    where
        T: Into<Text<'text>>,
    {
        self.push_blocks(layout, Some(t.into()).into_iter())
    }

    pub fn write<S>(&mut self, pos: P, width: u32, text: S)
    where
        S: Into<Cow<'text, str>>,
    {
        self.push(
            TextBlock {
                origin: pos,
                width,
            },
            Text {
                text: text.into(),
                style: Style::default(),
            },
        );
    }

    pub fn reset(&mut self) {
        self.parts.clear();
        self.blocks.clear();
        self.char_count = 0;
    }

    /// Returns an estimate of how many glyphs are in the buffer. This may be inaccurate due to eg
    /// ligatures and normalization.
    pub fn glyph_estimate(&self) -> usize {
        self.char_count
    }

    pub fn parts(&self) -> &[Text<'text>] {
        &self.parts[..]
    }

    /// Runs the layout algorithm on the most recently added block, and returns the amount of space
    /// used, and its `TextBlock` object.
    pub fn measure_area(
        &mut self,
        render: &mut Render,
        query: Point<f32>,
    ) -> (
        Point<f32>,
        &mut TextBlock<P>,
        Option<QueryResult>,
    ) {
        let &mut(ref range, ref mut layout) = self.blocks.last_mut().expect("measure_area called but there were no blocks");
        let mut layout_block = layout.to_block();
        let mut glyphs = Vec::new();
        let mut max_area = point(0.0, 0.0);
        let mut qr = None;
        for (part_index, text) in self.parts[range.clone()].iter().enumerate() {
            let (nm, q) = layout_block_glyphs(
                &mut layout_block,
                &mut glyphs,
                render.hidpi_factor,
                &render.fonts,
                text,
                query,
            );
            if let (None, Some(mut q)) = (&qr, q) {
                q.part_index = part_index;
                qr = Some(q);
            }
            if nm.x > max_area.x { max_area.x = nm.x; }
            if nm.y > max_area.y { max_area.y = nm.y; }
        }
        layout_block.bump_line(0.0);
        (max_area, layout, qr)
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
    pub fn build<F>(
        &mut self,
        render: &mut Render,
        mut write_glyph: F
    )
    where
        F: for<'x, 'y> FnMut(&'x Text<'y>, &P, Rect<i32>, Rect<f32>) -> FlowControl,
    {
        let mut glyphs: Vec<(usize, PositionedGlyph)> = Vec::new();
        let mut glyph_block: Vec<(&TextBlock<P>, &Text, Range<usize>)> = Vec::new();
        for (range, layout) in &self.blocks {
            let mut layout_block = layout.to_block();
            for text in &self.parts[range.clone()] {
                let start = glyphs.len();
                layout_block_glyphs(
                    &mut layout_block,
                    &mut glyphs,
                    render.hidpi_factor,
                    &render.fonts,
                    text,
                    point(-9e9, -9e9),
                );
                let end = glyphs.len();
                glyph_block.push((layout, text, start..end));
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
        for (layout, text, range) in glyph_block {
            for &(fontid, ref glyph) in &glyphs[range] {
                if let Ok(Some((uv, pos))) = render.cache.rect_for(fontid, glyph) {
                    match write_glyph(text, &layout.origin, pos, uv) {
                        FlowControl::NextGlyph => (),
                        FlowControl::SkipBlock => break,
                        FlowControl::StopBuild => return,
                    }
                }
            }
        }
    }
}
fn layout_block_glyphs<'layout, 'context, 'fonts, 'result, 'text>(
    layout: &'layout mut LayoutBlock,
    result: &'result mut Vec<(usize, PositionedGlyph<'fonts>)>,
    dpi: f64,
    fonts: &'context [Font<'fonts>],
    text: &'text Text<'text>,
    query: Point<f32>,
) -> (Point<f32>, Option<QueryResult>)
where
    'fonts: 'context,
{
    let mut query_result = None;
    let style = &text.style;
    let scale = Scale::uniform((text.style.scale as f64 * dpi) as f32);
    let v_metrics = fonts[0].v_metrics(scale); // FIXME: Various problems.
    let advance_height: f32 = v_metrics.ascent - v_metrics.descent + v_metrics.line_gap; // FIXME: max(this line)?
    let is_separator = |c: char| c == ' ';
    let caret2rect = |caret: Point<f32>| -> Rect<f32> {
        let min = caret - vector(0.0, v_metrics.ascent);
        let max = caret; // Tails on j's and g's shouldn't count, I think. - vector(0.0, v_metrics.descent);
        Rect { min, max }
    };
    layout.query_area = caret2rect(layout.caret);
    for (nfc_char_index, c) in text.text.nfc().enumerate() {
        if c.is_control() {
            if c == '\n' {
                layout.bump_line(advance_height);
                layout.query_area = caret2rect(layout.caret);
                layout.wrap_unit_start = result.len();
                layout.any_break = false;
                layout.query_hit = false;
            }
            continue;
        }
        if is_separator(c) {
            layout.wrap_unit_start = result.len() + 1; // We break at the *next* character
            layout.any_break = true;
            layout.query_hit = false;
            layout.query_area = caret2rect(layout.caret);
        }
        let (fontid, base_glyph) = style.get_glyph(c, fonts);
        let font = &fonts[fontid];
        if !layout.just_wrapped {
            if let Some(id) = layout.last_glyph_id.replace(base_glyph.id()) {
                layout.caret.x += font.pair_kerning(scale, id, base_glyph.id()); // FIXME: Not after a wrap.
            }
        }
        let mut glyph: PositionedGlyph = base_glyph.scaled(scale).positioned(layout.caret);
        let bb = glyph.pixel_bounding_box();
        if let Some(bb) = bb {
            if bb.max.x as i64 > layout.width as i64 {
                layout.bump_line(advance_height);
                if result.len() > layout.wrap_unit_start && layout.any_break {
                    // There's probably some weird degenerate case where this'd panic w/o this
                    // check.
                    let delta = layout.caret - result[layout.wrap_unit_start].1.position();
                    for &mut (_, ref mut g) in &mut result[layout.wrap_unit_start..] {
                        *g = g.clone().into_unpositioned().positioned(g.position() + delta);
                    }
                    let last = &result.last().expect("any glyphs").1;
                    layout.caret = last.position();
                    layout.caret.x += last.unpositioned().h_metrics().advance_width;
                    // Word-wrapping may cause us to either LOSE or GAIN the query, but we only
                    // need to check LOSE because a GAIN will natively re-check.
                    layout.query_area.min = layout.query_area.min + delta;
                    layout.query_area.max = layout.query_area.max + delta;
                } else {
                    // If there were glyphs, then everything must get shoved down.
                    // But if there were no glyphs, we still need to reset the query_area.
                    layout.query_area = caret2rect(layout.caret);
                }
                if layout.query_hit {
                    layout.query_hit = false;
                    query_result = None;
                }
                layout.any_break = false;
                glyph = glyph.into_unpositioned().positioned(layout.caret);
                layout.last_glyph_id = None;
                layout.just_wrapped = true;
            }
        }
        let metrics = glyph.unpositioned().h_metrics();
        layout.caret.x += metrics.advance_width;
        layout.query_area.max.x = layout.caret.x;
        if true
            && query_result.is_none()
            && (layout.query_area.min.x <= query.x && query.x <= layout.query_area.max.x)
            && (layout.query_area.min.y <= query.y && query.y <= layout.query_area.max.y)
            && layout.query_area.min.x != layout.query_area.max.x
        {
            layout.query_hit = true;
            query_result = Some(QueryResult {
                part_index: !0, // NOTE: Fixed later.
                nfc_char_index,
                area: layout.query_area,
            });
        } else if let (Some(qr), true) = (&mut query_result, layout.query_hit)  {
            qr.area.max.x = qr.area.max.x.max(layout.caret.x);
        }
        result.push((fontid, glyph));
        // FIXME: You might like to not push spaces, but they're load-bearing.
        layout.just_wrapped = false;
    }
    let mut ret = layout.max_area;
    ret.x = ret.x.max(layout.caret.x);
    (ret, query_result)
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    /// The text-part that the query found.
    pub part_index: usize,
    /// The character in the block found. **NOTE**: This is possibly not what you'd expect, since it
    /// is the index in the `str.nfc()` stream. Normalize your strings first.
    // Sorry, there doesn't seem to be any easy fix for this.
    pub nfc_char_index: usize,
    /// The area, on-screen, of the broken word that has been selected.
    /// Only useful for debugging.
    pub area: Rect<f32>,
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
                        color,
                    },
                    Vertex {
                        position: [gl_rect_min.x,  gl_rect_min.y],
                        tex_coords: [uv_rect.min.x, uv_rect.min.y],
                        color,
                    },
                    Vertex {
                        position: [gl_rect_max.x,  gl_rect_min.y],
                        tex_coords: [uv_rect.max.x, uv_rect.min.y],
                        color,
                    },
                    Vertex {
                        position: [gl_rect_max.x,  gl_rect_min.y],
                        tex_coords: [uv_rect.max.x, uv_rect.min.y],
                        color,
                    },
                    Vertex {
                        position: [gl_rect_max.x, gl_rect_max.y],
                        tex_coords: [uv_rect.max.x, uv_rect.max.y],
                        color,
                    },
                    Vertex {
                        position: [gl_rect_min.x, gl_rect_max.y],
                        tex_coords: [uv_rect.min.x, uv_rect.max.y],
                        color,
                    },
                ];
                vertices.extend_from_slice(&verts[..]);
                FlowControl::NextGlyph
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
