extern crate rusttyper;
extern crate glium;

use glium::{Surface};
use glium::glutin;
use rusttyper::{point, Point, Rect};

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
    let events_loop = glutin::event_loop::EventLoop::new();
    let window = glutin::window::WindowBuilder::new()
        .with_title("Rusttyper Example")
        .with_inner_size(glutin::dpi::LogicalSize::new(512.0, 512.0));
    let context = glutin::ContextBuilder::new()
        .with_vsync(true)
        .with_multisampling(8);
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    let mut font = rusttyper::Render::new(&display, 512, (0.1, 0.1));
    font.add_fonts(&include_bytes!("../DroidSans.ttf")[..]).unwrap();

    let uidraw = rusttyper::simple2d::Simple2d::create(&display).unwrap();
    let mut count = 0usize;
    let mut mouse = Point { x: 0.0, y: 0.0 };
    events_loop.run(move |event, _window, flow| {
        let mut stop = false;
        let mut draw = false;
        use glutin::event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode};
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => stop = true,
            Event::WindowEvent { event: WindowEvent::Destroyed, .. } => stop = true,
            Event::WindowEvent { event: WindowEvent::KeyboardInput { input: KeyboardInput { virtual_keycode: Some(VirtualKeyCode::Escape), .. }, .. }, .. } => stop = true,
            Event::WindowEvent { event: WindowEvent::CursorMoved { position, .. }, .. } => {
                mouse.x = position.x as f32;
                mouse.y = position.y as f32;
            },
            Event::RedrawRequested(_) => draw = true,
            Event::NewEvents(_) => {
                count += 1;
                display.gl_window().window().request_redraw();
                return;
            },
            _ => return,
        }
        if stop {
            *flow = glutin::event_loop::ControlFlow::Exit;
            return;
        } else {
            use std::time::*;
            *flow = glutin::event_loop::ControlFlow::WaitUntil(Instant::now() + Duration::from_secs_f64(1.0 / 30.0));
        }

        if !draw { return; }

        let mut buffer = ::rusttyper::RunBuffer::new();
        {
            let (screen_width, screen_height) = display.get_framebuffer_dimensions();
            let (screen_width, screen_height) = (screen_width as f32, screen_height as f32);
            let margin = 0.1;
            let input_field_width = (screen_width - 2.0 * margin * screen_width) as u32;
            let start = ((screen_width * margin) as i32, (screen_height * margin) as i32);
            use rusttyper::{Style, TextBlock};
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

            buffer.push_blocks(TextBlock {
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
            /*buffer.push_blocks(TextBlock {
                origin: start,
                width: input_field_width,
            }, vec!["hey_guys".into()].into_iter());*/
            {
                let query = point(mouse.x - start.0 as f32, mouse.y - start.1 as f32);
                let query = buffer.measure_area(&mut font, query);
                if let Some(query) = query.2 {
                    buffer.write((mouse.x as i32, mouse.y as i32), input_field_width, "@");
                    let area = query.area;
                    let area = Rect {
                        min: point((area.min.x + start.0 as f32) as i32, (area.min.y + start.1 as f32) as i32),
                        max: point((area.max.x + start.0 as f32) as i32, (area.max.y + start.1 as f32) as i32),
                    };
                    buffer.write((area.min.x, area.min.y), input_field_width, "@");
                    buffer.write((area.max.x, area.max.y), input_field_width, "@");
                    /*{
                        let bit = buffer.parts.get(query.run_index)
                            .map(|bit| -> &str { &*bit.text })
                            .unwrap_or("<WRONG RUN>")
                        ;
                        let i = query.char_index;
                        let lo = if i > 5 {
                            i - 5
                        } else {
                            i
                        };
                        let hi = if i + 5 < bit.len() {
                            i + 5
                        } else {
                            bit.len()
                        };
                        println!("{} {} {:?}  {:?}", query.run_index, query.char_index, bit.get(query.char_index..query.char_index+1).unwrap_or("><"), &bit.get(lo..hi).unwrap_or("><"));
                    }*/
                }
            }
            buffer.write((0, screen_height as i32), input_field_width, format!("{}", count));
        }

        let mut target = display.draw();
        target.clear_color(1.0, 1.0, 1.0, 0.0);
        uidraw.draw(&display, &mut target, &mut font, &mut buffer).unwrap();
        target.finish().unwrap();
    });
}

