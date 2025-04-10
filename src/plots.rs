pub use plotly;
pub use plotly::{
    color::{NamedColor, Rgb},
    common::{DashType, ExponentFormat, Font, Line, Marker, MarkerSymbol, Mode},
    layout::{Axis, Layout, Margin},
    ImageFormat, Plot, Scatter,
};
pub(crate) use std::process::Command;

#[derive(Clone, Copy, Debug)]
pub enum Format {
    PNG,
    JPEG,
    WEBP,
    SVG,
    PDF,
    EPS,
}

impl Into<ImageFormat> for Format {
    fn into(self) -> ImageFormat {
        match self {
            Format::PNG => ImageFormat::PNG,
            Format::JPEG => ImageFormat::JPEG,
            Format::WEBP => ImageFormat::WEBP,
            Format::SVG => ImageFormat::SVG,
            Format::PDF => ImageFormat::PDF,
            Format::EPS => ImageFormat::EPS,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PlotOptions {
    pub format: Format,
    pub width: usize,
    pub height: usize,
    pub scale: f64,
}

pub fn show(plot: &Plot, format: ImageFormat, options: PlotOptions) {
    let temp = "tempimages/temp";
    let file = format!("{temp}.{format}");
    plot.write_image(temp, format, options.width, options.height, options.scale);

    Command::new("wslview")
        .arg(&file)
        .output()
        .expect("vecio, sei messo male");
}

pub fn show_png(plot: &Plot, options: PlotOptions) {
    show(plot, ImageFormat::PNG, options)
}

pub fn show_svg(plot: &Plot, options: PlotOptions) {
    show(plot, ImageFormat::SVG, options)
}
