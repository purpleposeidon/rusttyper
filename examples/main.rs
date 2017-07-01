extern crate rusttyper;
extern crate glium;

use glium::{DisplayBuild, Surface};
use glium::glutin;

static TEXT: &'static str = r#"Hello, world!

We've got WORD WRAPPING...

Dolore velit iure culpa ea qui in distinctio. Aspernatur ipsa totam consequatur tempora doloribus. Nulla necessitatibus cumque non. Et illum fuga quae est. Eius ut dolorem officiis. Sint placeat facilis eos. Ut dolorem natus velit non maxime neque minima. Voluptatum vero architecto quis aut maiores quaerat facilis adipisci. Animi aut architecto odit et cumque praesentium earum. Praesentium aut sit distinctio voluptates odio sed quos. Consequuntur nostrum est voluptatem ducimus saepe officiis. Dolores ut ut maiores. Dolore molestiae aut nam rem perferendis est dolorum sint. Doloremque laboriosam reprehenderit unde et architecto.


We do an okay job handling LONG WORDS...

pneumonoultramicroscopicsilicovolcanoconiosis supercalifragilisticexpialidocious siebenhundertsiebenundsiebzigtausendsiebenhundertsiebenundsiebzig


But we don't yet have support for FALLBACK FONTS!

ウィキペディアへようこそ ウィキペディアは誰でも編集できるフリー百科事典です

Nor can we handle LTR text!?

السير ونستون ليونارد سبنسر تشرشل هو رئيس وزراء المملكة المتحدة من سنة 1940 وحتى سنة 1945 (إبان الحرب العالمية الثانية)، ومن سنة 1951 إلى سنة 1955.

"#;

fn main() {
    let display = glutin::WindowBuilder::new()
        .with_vsync()
        .with_dimensions(512, 512)
        .with_title("Rusttyper Example")
        .build_glium()
        .unwrap();

    let mut font = rusttyper::Render::new(&display, 512, (0.1, 0.1));
    font.add_fonts(&include_bytes!("../DroidSans.ttf")[..]);

    let uidraw = rusttyper::simple2d::Simple2d::create(&display).unwrap();
    let mut count = 0usize;
    loop {
        count += 1;
        for event in display.poll_events() {
            match event {
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::Closed => return,
                _ => {}
            }
        }

        let mut buffer = ::rusttyper::RunBuffer::new();
        {
            let (screen_width, screen_height) = display.get_framebuffer_dimensions();
            let (screen_width, screen_height) = (screen_width as f32, screen_height as f32);
            let margin = 0.1;
            let input_field_width = (screen_width - 2.0 * margin * screen_width) as u32;
            let start = ((screen_width * margin) as i32, (screen_height * margin) as i32);
            use rusttyper::{Style, Layout};
            let m = |n| (count * n % 255) as u8;
            let colorful = Style {
                color: (m(5), m(16), m(32), 255),
                .. Style::default()
            };
            let big = Style {
                scale: 32.0,
                .. Style::default()
            };
            let small = Style {
                scale: 10.0,
                .. Style::default()
            };
            buffer.push_run(Layout {
                origin: start,
                width: input_field_width,
            }, vec![
                TEXT.into(),
                "\n...Did I mention that we have ".into(),
                (colorful, "COLORS").into(),
                "?".into(),
                "\nAnd also ".into(),
                (big, "FONT SIZES").into(),
                "?".into(),
                (small, " (okay that might be too big)").into(),
            ].into_iter());
            buffer.write((0, screen_height as i32), input_field_width, format!("{}", count));
        }

        let mut target = display.draw();
        target.clear_color(1.0, 1.0, 1.0, 0.0);
        uidraw.draw(&display, &mut target, &mut font, &mut buffer).unwrap();
        target.finish().unwrap();
    }
}

