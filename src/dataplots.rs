use crate::utils::*;
use libc::TIME_OOP;
use polars::prelude::*;
use std::fs::File;

pub fn plot(path: &str, name: &str, show: bool) -> SolverResult<()> {
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_try_parse_dates(true))
        .try_into_reader_with_file_path(Some(path.into()))
        .expect("Could not load file")
        .finish()
        .expect("Could not finish");

    let t_out = get_column(&df, "time");
    // let avg_free_fidelity = get_column(&df, "avg_free_fidelity");
    // let avg_ideal_fidelity = get_column(&df, "avg_ideal_fidelity");
    // let avg_ctrl_fidelity = get_column(&df, "avg_ctrl_fidelity");
    // let avg_time_fidelity1 = get_column(&df, "avg_time_fidelity1");
    // let avg_time_fidelity2 = get_column(&df, "avg_time_fidelity2");
    // let avg_time_fidelity3 = get_column(&df, "avg_time_fidelity3");
    // let avg_time_fidelity4 = get_column(&df, "avg_time_fidelity4");

    let mut avg_free_min_purity = get_column(&df, "avg_free_min_purity");
    let mut avg_ideal_min_purity = get_column(&df, "avg_ideal_min_purity");
    let mut avg_ctrl_min_purity = get_column(&df, "avg_ctrl_min_purity");
    let mut avg_time_min_purity1 = get_column(&df, "avg_time_min_purity1");
    let mut avg_time_min_purity2 = get_column(&df, "avg_time_min_purity2");
    let mut avg_time_min_purity3 = get_column(&df, "avg_time_min_purity3");
    let mut avg_time_min_purity4 = get_column(&df, "avg_time_min_purity4");

    let mut plot = plotpy::Plot::new();

    // let mut free_curve = plotpy::Curve::new();
    // free_curve
    //     .set_label("Free evolution")
    //     .draw(&t_out, &avg_free_fidelity);
    //
    // let mut ideal_curve = plotpy::Curve::new();
    // ideal_curve
    //     .set_label("Ideal evolution")
    //     .draw(&t_out, &avg_ideal_fidelity);
    //
    // let mut ctrl_curve = plotpy::Curve::new();
    // ctrl_curve
    //     .set_label("Controlled evolution")
    //     .draw(&t_out, &avg_ctrl_fidelity);
    //
    // let mut time_curve1 = plotpy::Curve::new();
    // time_curve1
    //     .set_label(&format!("Windowed evolution, $k = {}$", 5000))
    //     .draw(&t_out, &avg_time_fidelity1);
    //
    // let mut time_curve2 = plotpy::Curve::new();
    // time_curve2
    //     .set_label(&format!("Windowed evolution, $k = {}$", 20000))
    //     .draw(&t_out, &avg_time_fidelity2);
    //
    // let mut time_curve3 = plotpy::Curve::new();
    // time_curve3
    //     .set_label(&format!("Windowed evolution, $k = {}$", 50000))
    //     .draw(&t_out, &avg_time_fidelity3);
    //
    // let mut time_curve4 = plotpy::Curve::new();
    // time_curve4
    //     .set_label(&format!("Windowed evolution, $k = {}$", 100000))
    //     .draw(&t_out, &avg_time_fidelity4);

    let mut free_purity_curve = plotpy::Curve::new();
    free_purity_curve
        .set_label("Free evolution")
        .draw(&t_out, &avg_free_min_purity);

    let mut ideal_purity_curve = plotpy::Curve::new();
    ideal_purity_curve
        .set_label("Ideal evolution")
        .draw(&t_out, &avg_ideal_min_purity);

    let mut ctrl_purity_curve = plotpy::Curve::new();
    ctrl_purity_curve
        .set_label("Controlled evolution")
        .draw(&t_out, &avg_ctrl_min_purity);

    let mut time_purity_curve1 = plotpy::Curve::new();
    time_purity_curve1
        .set_label(&format!("Windowed evolution, $k = {}$", 5000))
        .draw(&t_out, &avg_time_min_purity1);

    let mut time_purity_curve2 = plotpy::Curve::new();
    time_purity_curve2
        .set_label(&format!("Windowed evolution, $k = {}$", 20000))
        .draw(&t_out, &avg_time_min_purity2);

    let mut time_purity_curve3 = plotpy::Curve::new();
    time_purity_curve3
        .set_label(&format!("Windowed evolution, $k = {}$", 50000))
        .draw(&t_out, &avg_time_min_purity3);

    let mut time_purity_curve4 = plotpy::Curve::new();
    time_purity_curve4
        .set_label(&format!("Windowed evolution, $k = {}$", 100000))
        .draw(&t_out, &avg_time_min_purity4);

    plot.extra("plt.rcParams.update({\"text.usetex\": True, \"font.family\": \"Helvetica\"})\n")
        .extra("plt.rcParams['figure.constrained_layout.use'] = True\n")
        .set_save_tight(true)
        .add(&free_purity_curve)
        .add(&ideal_purity_curve)
        .add(&ctrl_purity_curve)
        .add(&time_purity_curve1)
        .add(&time_purity_curve2)
        .add(&time_purity_curve3)
        .add(&time_purity_curve4)
        // .add(&free_curve)
        // .add(&ideal_curve)
        // .add(&ctrl_curve)
        // .add(&time_curve1)
        // .add(&time_curve2)
        // .add(&time_curve3)
        // .add(&time_curve4)
        // .set_labels("$t$", r"$F(\rho_t)$")
        .set_labels("$t$", r"Min purity")
        .legend();

    if show {
        plot.show(&format!("./Images/{}.svg", name))?;
    } else {
        plot.save(&format!("./Images/{}.svg", name))?;
    }
    Ok(())
}

fn get_column(df: &DataFrame, name: &str) -> Vec<f64> {
    df[name]
        .f64()
        .unwrap()
        .iter()
        .collect::<Option<Vec<f64>>>()
        .unwrap()
}
