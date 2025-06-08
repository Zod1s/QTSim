// pub use plotly;
// pub use plotly::{
//     color::{NamedColor, Rgb},
//     common::{DashType, ExponentFormat, Font, Line, Marker, MarkerSymbol, Mode},
//     layout::{Axis, Layout, Margin},
//     ImageFormat, Plot, Scatter,
// };

use nalgebra as na;
// use plotters::prelude::*;
use plotpy;
pub(crate) use std::process::Command;

pub fn plot_obsv(ts: &Vec<f64>, obsv: &Vec<na::SVector<f64, 3>>) -> Result<(), plotpy::StrError> {
    // let root = BitMapBackend::new("plotters/obsv.png", (600, 600)).into_drawing_area();
    // root.fill(&WHITE)?;
    //
    // let mut chart = ChartBuilder::on(&root)
    //     .caption("Observable evolution", ("sans-serif", 50).into_font())
    //     .margin(5)
    //     .x_label_area_size(30)
    //     .y_label_area_size(30)
    //     .build_cartesian_2d(0f32..ts[ts.len() - 1] as f32, -1f32..1f32)?;
    //
    // chart.configure_mesh().draw()?;
    //
    // chart
    //     .draw_series(LineSeries::new(
    //         ts.iter()
    //             .zip(obsv)
    //             .map(|(t, o)| (*t as f32, o[0] as f32))
    //             .collect::<Vec<(f32, f32)>>(),
    //         &RED,
    //     ))?
    //     .label("X observable")
    //     .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    //
    // chart
    //     .draw_series(LineSeries::new(
    //         ts.iter()
    //             .zip(obsv)
    //             .map(|(t, o)| (*t as f32, o[1] as f32))
    //             .collect::<Vec<(f32, f32)>>(),
    //         &GREEN,
    //     ))?
    //     .label("Y observable")
    //     .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));
    //
    // chart
    //     .draw_series(LineSeries::new(
    //         ts.iter()
    //             .zip(obsv)
    //             .map(|(t, o)| (*t as f32, o[2] as f32))
    //             .collect::<Vec<(f32, f32)>>(),
    //         &BLUE,
    //     ))?
    //     .label("Z observable")
    //     .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    //
    // chart
    //     .configure_series_labels()
    //     .background_style(&WHITE.mix(0.8))
    //     .border_style(&BLACK)
    //     .draw()?;
    //
    // root.present()?;

    let mut xobs = plotpy::Curve::new();
    xobs.set_label("X observable")
        .set_line_color("#FF0000")
        .set_line_width(3.0)
        .set_line_style("-");

    xobs.draw(ts, &obsv.iter().map(|o| o[0]).collect::<Vec<f64>>());

    let mut yobs = plotpy::Curve::new();
    yobs.set_label("Y observable")
        .set_line_color("#00FF00")
        .set_line_width(3.0)
        .set_line_style("--");

    yobs.draw(ts, &obsv.iter().map(|o| o[1]).collect::<Vec<f64>>());

    let mut zobs = plotpy::Curve::new();
    zobs.set_label("Z observable")
        .set_line_color("#0000FF")
        .set_line_width(3.0)
        .set_line_style("-");

    zobs.draw(ts, &obsv.iter().map(|o| o[2]).collect::<Vec<f64>>());

    let mut plot = plotpy::Plot::new();
    plot.add(&xobs)
        .add(&yobs)
        .add(&zobs)
        .set_range(0.0, 2.0, -1.0, 1.0)
        .grid_labels_legend("time", "observable value");

    plot.show("tempimages")
}

// #[derive(Clone, Copy, Debug)]
// pub enum Format {
//     PNG,
//     JPEG,
//     WEBP,
//     SVG,
//     PDF,
//     EPS,
// }
//
// impl Into<ImageFormat> for Format {
//     fn into(self) -> ImageFormat {
//         match self {
//             Format::PNG => ImageFormat::PNG,
//             Format::JPEG => ImageFormat::JPEG,
//             Format::WEBP => ImageFormat::WEBP,
//             Format::SVG => ImageFormat::SVG,
//             Format::PDF => ImageFormat::PDF,
//             Format::EPS => ImageFormat::EPS,
//         }
//     }
// }
//
// #[derive(Clone, Copy, Debug)]
// pub struct PlotOptions {
//     pub format: Format,
//     pub width: usize,
//     pub height: usize,
//     pub scale: f64,
// }
//
// pub fn show(plot: &Plot, format: ImageFormat, options: PlotOptions) {
//     let temp = "tempimages/temp";
//     let file = format!("{temp}.{format}");
//     plot.write_image(temp, format, options.width, options.height, options.scale);
//
//     Command::new("wslview")
//         .arg(&file)
//         .output()
//         .expect("vecio, sei messo male");
// }
//
// pub fn show_png(plot: &Plot, options: PlotOptions) {
//     show(plot, ImageFormat::PNG, options)
// }
//
// pub fn show_svg(plot: &Plot, options: PlotOptions) {
//     show(plot, ImageFormat::SVG, options)
// }
