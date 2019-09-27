import com.opencsv.CSVReader;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.category.LineAndShapeRenderer;
import org.jfree.chart.renderer.xy.StandardXYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.chart.util.ShapeUtils;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class KNN {

    public static double calculateFMeasureForSetup(double[][] objects, int[] classes, int classesNum,
                                                   String distance, String kernel, String window, double windowSize) {
        int[][] cm = new int[classesNum][classesNum];
        for (int i = 0; i < objects.length; i++) {
            int res = Utils.learnAndAnswerQuery(objects, classes, i, classesNum, distance, kernel, window, windowSize);
            if (res == 0) return -1;
            cm[res - 1][classes[i] - 1]++;
        }
        return Utils.fmeasure(cm);
    }

    public static void main(String[] args) throws IOException {
        BufferedWriter log = new BufferedWriter(new FileWriter("log.txt"));
        List<String[]> values = new CSVReader(new FileReader("src/main/resources/dataset_16_mfeat-karhunen.csv")).readAll();
        values.remove(0);  // header
        int samples = values.size();
        int attributes = values.get(0).length - 1;
        double[][] objects = new double[samples][attributes];
        int[] classes = new int[samples];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < attributes; j++) {
                objects[i][j] = Double.valueOf(values.get(i)[j]);
            }
            classes[i] = Integer.valueOf(values.get(i)[attributes]);
        }
        int classesNum = (int)Arrays.stream(classes).distinct().count();
        for (int i = 0; i < attributes; i++) {
            double max = 0;
            for (int j = 0; j < samples; j++) {
                if (max < Math.abs(objects[j][i])) max = Math.abs(objects[j][i]);
            }
            for (int j = 0; j < samples; j++) {
                objects[j][i] /= max;
            }
        }
        String[] distances = new String[] {
                  "euclidean"
                , "manhattan"
                , "chebyshev"
        };
        String[] kernels = new String[] {
                  "uniform"
                , "triangular"
                , "epanechnikov"
                , "quartic"
                , "triweight"
                , "tricube"
                , "gaussian"
                , "cosine"
                , "logistic"
                , "sigmoid"
        };
        double max = 0;
        String maxSetup = "";
        for (String distance: distances) {
            for (String kernel: kernels) {
                XYSeriesCollection ds = new XYSeriesCollection();
                XYSeries series = new XYSeries("");
                double lower = 0;
                double upper = 0;
                double step = 0;
                int neighbors = 0;
                switch (distance) {
                    case "euclidean": {
                        lower = 0.02;
                        upper = 3.5;
                        if (kernel.equals("gaussian") || kernel.equals("logistic") || kernel.equals("sigmoid")) upper = 1;
                        step = 0.02;
                        neighbors = 60;
                        break;
                    }
                    case "manhattan": {
                        lower = 15;
                        if (kernel.equals("gaussian") || kernel.equals("logistic") || kernel.equals("sigmoid")) lower = 0.05;
                        upper = 20;
                        if (kernel.equals("gaussian") || kernel.equals("logistic") || kernel.equals("sigmoid")) upper = 6;
                        step = 0.05;
                        neighbors = 80;
                        break;
                    }
                    case "chebyshev": {
                        lower = 0.01;
                        upper = 1.5;
                        if (kernel.equals("gaussian") || kernel.equals("logistic") || kernel.equals("sigmoid")) upper = 1;
                        step = 0.01;
                        neighbors = 60;
                        break;
                    }
                }
                for (double window = lower; window < upper; window += step) {
                    double f = calculateFMeasureForSetup(objects, classes, classesNum, distance, kernel, "fixed", window);
                    if (f == -1.0) continue;
                    if (f < 0.75) {
                        log.write("For " + distance + " distance, " + kernel + " kernel, fixed window of "
                                + String.format("%.4f", window) + ": F1 measure is " + String.format("%.6f", f) + "\n");
                        break;
                    }
                    if (f > max) {
                        max = f;
                        maxSetup = distance + "-" + kernel + "-fixed-" + window;
                    }
                    series.add(window, f);
                    log.write("For " + distance + " distance, " + kernel + " kernel, fixed window of "
                            + String.format("%.4f", window) + ": F1 measure is " + String.format("%.6f", f) + "\n");
                }
                log.flush();
                ds.addSeries(series);
                JFreeChart ch = ChartFactory.createXYLineChart(distance + "-" + kernel + "-fixed",
                        "Window size", "F1 measure", ds, PlotOrientation.VERTICAL, false, false, false);
                final XYPlot plot = ch.getXYPlot();
                XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
                renderer.setSeriesPaint(0, Color.RED);
                renderer.setSeriesStroke(0, new BasicStroke(4.0f));
                plot.setBackgroundPaint(Color.white);
                plot.setRangeGridlinePaint(Color.black);
                plot.setRangeGridlinesVisible(true);
                plot.setDomainGridlinePaint(Color.black);
                plot.setDomainGridlinesVisible(true);
                plot.setRenderer(renderer);
                try {
                    OutputStream out = new FileOutputStream(ch.getTitle().getText() + ".png");
                    ChartUtils.writeChartAsPNG(out, ch, 1280, 720);
                    out.close();
                } catch (IOException ex) {
                }


                XYSeriesCollection ds1 = new XYSeriesCollection();
                XYSeries series1 = new XYSeries("");
                for (int k = 1; k < neighbors; k++) {
                    double f = calculateFMeasureForSetup(objects, classes, classesNum, distance, kernel, "variable", k);
                    if (f == -1.0) continue;
                    if (f < 0.75) {
                        log.write("For " + distance + " distance, " + kernel + " kernel, variable window of " + k
                                + " neighbors: F1 measure is " + String.format("%.6f", f) + "\n");
                        break;
                    }
                    if (f > max) {
                        max = f;
                        maxSetup = distance + "-" + kernel + "-variable-" + k;
                    }
                    series1.add(k, f);
                    log.write("For " + distance + " distance, " + kernel + " kernel, variable window of " + k
                            + " neighbors: F1 measure is " + String.format("%.6f", f) + "\n");
                }
                log.flush();
                ds1.addSeries(series1);
                JFreeChart ch1 = ChartFactory.createXYLineChart(distance + "-" + kernel + "-variable",
                        "Neighbors", "F1 measure", ds1, PlotOrientation.VERTICAL, false, false, false);
                final XYPlot plot1 = ch1.getXYPlot();
                plot1.setBackgroundPaint(Color.white);
                plot1.setRangeGridlinePaint(Color.black);
                plot1.setRangeGridlinesVisible(true);
                plot1.setDomainGridlinePaint(Color.black);
                plot1.setDomainGridlinesVisible(true);
                plot1.setRenderer(renderer);
                try {
                    OutputStream out = new FileOutputStream(ch1.getTitle().getText() + ".png");
                    ChartUtils.writeChartAsPNG(out, ch1, 1280, 720);
                    out.close();
                } catch (IOException ex) {
                }
            }
        }
        log.write("Optimal combination: " + maxSetup + ": " + max + "\n");
        log.flush();
    }

}
