package ml;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Utils {

    public static int nextInt(BufferedReader sc) throws IOException {
        int a = 0;
        int k = 1;
        int b = sc.read();
        while (b < '0' || b > '9') {
            if (b == '-') k = -1;
            b = sc.read();
        }
        while (b >= '0' && b <= '9') {
            a = a * 10 + (b - '0');
            b = sc.read();
        }
        return a * k;
    }

    public static double euclidean(double[] fst, double[] snd) {
        double res = 0;
        for (int i = 0; i < fst.length; i++) {
            res += (fst[i] - snd[i]) * (fst[i] - snd[i]);
        }
        return Math.sqrt(res);
    }

    public static double manhattan(double[] fst, double[] snd) {
        double res = 0;
        for (int i = 0; i < fst.length; i++) {
            res += Math.abs(fst[i] - snd[i]);
        }
        return res;
    }

    public static double chebyshev(double[] fst, double[] snd) {
        double res = 0;
        for (int i = 0; i < fst.length; i++) {
            double t = Math.abs(fst[i] - snd[i]);
            if (res < t) res = t;
        }
        return res;
    }

    public static double uniform(double val) {
        if (Math.abs(val) < 1) return 1D / 2;
        else return 0;
    }

    public static double triangular(double val) {
        if (Math.abs(val) < 1) return 1D - Math.abs(val);
        else return 0;
    }

    public static double epanechnikov(double val) {
        double t = (1 - val * val);
        if (Math.abs(val) < 1) return 3 * t / 4;
        else return 0;
    }

    public static double quartic(double val) {
        double t = (1 - val * val);
        if (Math.abs(val) < 1) return 15 * t * t / 16;
        else return 0;
    }

    public static double triweight(double val) {
        double t = (1 - val * val);
        if (Math.abs(val) < 1) return 35 * t * t * t / 32;
        else return 0;
    }

    public static double tricube(double val) {
        double t = (1 - Math.abs(val) * val * val);
        if (Math.abs(val) < 1) return 70 * t * t * t / 81;
        else return 0;
    }

    public static double gaussian(double val) {
        return 1 / Math.sqrt(2 * Math.PI) * Math.exp(- val * val / 2);
    }

    public static double cosine(double val) {
        if (Math.abs(val) < 1) return Math.PI * ((long)(Math.pow(10, 9) * Math.cos(Math.PI * val / 2)) / Math.pow(10, 9)) / 4;
        else return 0;
    }

    public static double logistic(double val) {
        return 1 / (Math.exp(val) + 2 + Math.exp(- val));
    }

    public static double sigmoid(double val) {
        return 2 / (Math.PI * (Math.exp(val) + Math.exp(- val)));
    }

    public static double fmeasure(int[][] cm) {
        int k = cm.length;
        int totalSum = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                totalSum += cm[i][j];
            }
        }
        double precision = 0;
        double recall = 0;
        for (int i = 0; i < k; i++) {
            int fp = 0;
            for (int j = 0; j < k; j++) {
                if (i != j) fp += cm[i][j];
            }
            double weight = (double)(fp + cm[i][i])/totalSum;
            int fn = 0;
            for (int j = 0; j < k; j++) {
                if (i != j) fn += cm[j][i];
            }
            double pr = (cm[i][i] + fp == 0) ? 0 : (double)cm[i][i]/(cm[i][i] + fp);
            double re = (cm[i][i] + fn == 0) ? 0 : (double)cm[i][i]/(cm[i][i] + fn);
            precision += pr * weight;
            recall += re * weight;
        }
        return 2 * (precision * recall) / (precision + recall);
    }

    public static double fmeasureMicro(int[][] cm) {
        int k = cm.length;
        int totalSum = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                totalSum += cm[i][j];
            }
        }
        double fT = 0;
        for (int i = 0; i < k; i++) {
            int fp = 0;
            for (int j = 0; j < k; j++) {
                if (i != j) fp += cm[i][j];
            }
            double weight = (double)(fp + cm[i][i])/totalSum;
            int fn = 0;
            for (int j = 0; j < k; j++) {
                if (i != j) fn += cm[j][i];
            }
            double pr = (cm[i][i] + fp == 0) ? 0 : (double)cm[i][i]/(cm[i][i] + fp);
            double re = (cm[i][i] + fn == 0) ? 0 : (double)cm[i][i]/(cm[i][i] + fn);
            double f = (pr + re == 0) ? 0 : (2 * (pr * re)/(pr + re));
            fT += f * weight;
        }
        return fT;
    }

    public static double scalarProduct(double[] fst, double[] snd) {
        double res = 0;
        for (int i = 0; i < fst.length; i++) res += fst[i] * snd[i];
        return res;
    }

    public static double[][] multiply(double[][] fst, double[][] snd) {
        double[][] res = new double[fst.length][snd[0].length];
        double[][] sndTransposed = transpose(snd);
        for (int i = 0; i < fst.length; i++) {
            double[] aFst = fst[i];
            for (int j = 0; j < snd[0].length; j++) {
                res[i][j] = scalarProduct(aFst, sndTransposed[j]);
            }
        }
        return res;
    }

    public static double[][] transpose(double[][] matrix) {
        double[][] res = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                res[i][j] = matrix[j][i];
            }
        }
        return res;
    }

    public static double[] subtract(double[] fst, double[] snd) {
        double[] res = new double[fst.length];
        for (int i = 0; i < fst.length; i++) res[i] = fst[i] - snd[i];
        return res;
    }

    public static double norm(double[] vec) {
        double res = 0;
        for (int i = 0; i < vec.length; i++) res += vec[i] * vec[i];
        return Math.sqrt(res);
    }

    public static void writeChartForDS(String chartName, XYSeriesCollection ds, String xName, String yName) {
        JFreeChart ch = ChartFactory.createXYLineChart(chartName,
                xName, yName, ds, PlotOrientation.VERTICAL, true, false, false);
        final XYPlot plot = ch.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesPaint(1, Color.BLUE);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        renderer.setSeriesStroke(1, new BasicStroke(2.0f));
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
    }

    public static <T> List<T> mergeExcept(List<List<T>> lists, int... ex) {
        List<T> res = new ArrayList<>();
        List<Integer> listEx = Arrays.stream(ex).boxed().collect(Collectors.toList());
        for (int i = 0; i < lists.size(); i++) {
            if (!listEx.contains(i)) res.addAll(lists.get(i));
        }
        return res;
    }

}
