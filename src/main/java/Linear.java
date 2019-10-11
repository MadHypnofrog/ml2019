import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;

public class Linear {

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

    public static double nextDouble(BufferedReader sc) throws IOException {
        String s = "";
        int b = sc.read();
        while (b < '0' || b > '9') {
            b = sc.read();
        }
        while ((b >= '0' && b <= '9') || b == '.') {
            s += (char)b;
            b = sc.read();
        }
        return Double.parseDouble(s);
    }

    private static double scalarProduct(double[] fst, double[] snd) {
        double res = 0;
        for (int i = 0; i < fst.length; i++) res += fst[i] * snd[i];
        return res;
    }

    private static double[][] multiply(double[][] fst, double[][] snd) {
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

    private static double[][] transpose(double[][] matrix) {
        double[][] res = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                res[i][j] = matrix[j][i];
            }
        }
        return res;
    }

    public static double[] calcGradientDescent(String fileName, int num) throws IOException {
        BufferedReader r = new BufferedReader(new FileReader(new File(fileName)));
        int m = nextInt(r);
        int n = nextInt(r);
        double[][] matrix = new double[n][m + 1];
        double[][] target = new double[n][1];
        double maxY = -10000000;
        double minY = 10000000;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                matrix[i][j] = nextDouble(r);
            }
            matrix[i][m] = 1;
            target[i][0] = nextDouble(r);
            if (maxY < target[i][0]) maxY = target[i][0];
            if (minY > target[i][0]) minY = target[i][0];
        }
        int testN = nextInt(r);
        double[][] testMatrix = new double[testN][m + 1];
        double[][] testTarget = new double[testN][1];
        double maxYTest = -10000000;
        double minYTest = 10000000;
        for (int i = 0; i < testN; i++) {
            for (int j = 0; j < m; j++) {
                testMatrix[i][j] = nextDouble(r);
            }
            testMatrix[i][m] = 1;
            testTarget[i][0] = nextDouble(r);
            if (maxYTest < testTarget[i][0]) maxYTest = testTarget[i][0];
            if (minYTest > testTarget[i][0]) minYTest = testTarget[i][0];
        }
        double[][] left = multiply(transpose(matrix), matrix);
        double[] right = transpose(multiply(transpose(matrix), target))[0];
        double[][] res = new double[m + 1][1];
        for (int i = 0; i < m + 1; i++) res[i][0] = 1;
        XYSeriesCollection ds = new XYSeriesCollection();
        XYSeriesCollection ds2 = new XYSeriesCollection();
        XYSeries series = new XYSeries("1");
        XYSeries series2 = new XYSeries("2");
        for (int iter = 0; iter < 1000; iter++) {
            double[][] grad = multiply(left, res);
            double norm = 0;
            for (int i = 0; i < m + 1; i++) {
                grad[i][0] -= right[i];
                norm += grad[i][0] * grad[i][0];
            }
            if (norm < 0.0000000000000000001) break;
            double[][] grad2 = multiply(left, grad);
            double a = 0;
            double b = 0;
            for (int i = 0; i < m + 1; i++) {
                a += grad[i][0] * grad[i][0];
                b += grad2[i][0] * grad[i][0];
            }
            for (int i = 0; i < m + 1; i++) {
                res[i][0] -= a * grad[i][0] / b;
            }
            double q = 0;
            double[] tR = transpose(res)[0];
            for (int i = 0; i < n; i++) {
                q += (scalarProduct(matrix[i], tR) - target[i][0]) * (scalarProduct(matrix[i], tR) - target[i][0]);
            }
            series.add(iter + 1, Math.sqrt(q) * (maxY - minY));
            double qTest = 0;
            for (int i = 0; i < testN; i++) {
                qTest += (scalarProduct(testMatrix[i], tR) - testTarget[i][0]) * (scalarProduct(testMatrix[i], tR) - testTarget[i][0]);
            }
            series2.add(iter + 1, Math.sqrt(qTest) / (maxYTest - minYTest));
        }
        ds.addSeries(series);
        ds2.addSeries(series2);
        JFreeChart ch = ChartFactory.createXYLineChart("Q(w) to Iterations (training set " + num + ")",
                "Iterations", "Q(w)", ds, PlotOrientation.VERTICAL, false, false, false);
        JFreeChart ch2 = ChartFactory.createXYLineChart("Q(w) to Iterations (test set " + num + ")",
                "Iterations", "Q(w)", ds2, PlotOrientation.VERTICAL, false, false, false);
        final XYPlot plot = ch.getXYPlot();
        final XYPlot plot2 = ch2.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        plot.setBackgroundPaint(Color.white);
        plot.setRangeGridlinePaint(Color.black);
        plot.setRangeGridlinesVisible(true);
        plot.setDomainGridlinePaint(Color.black);
        plot.setDomainGridlinesVisible(true);
        plot2.setBackgroundPaint(Color.white);
        plot2.setRangeGridlinePaint(Color.black);
        plot2.setRangeGridlinesVisible(true);
        plot2.setDomainGridlinePaint(Color.black);
        plot2.setDomainGridlinesVisible(true);
        plot.setRenderer(renderer);
        plot2.setRenderer(renderer);
        try {
            OutputStream out = new FileOutputStream(ch.getTitle().getText() + ".png");
            ChartUtils.writeChartAsPNG(out, ch, 1280, 720);
            out.close();
        } catch (IOException ex) {
        }
        try {
            OutputStream out = new FileOutputStream(ch2.getTitle().getText() + ".png");
            ChartUtils.writeChartAsPNG(out, ch2, 1280, 720);
            out.close();
        } catch (IOException ex) {
        }
        return transpose(res)[0];
    }

    private static double[] solveGauss(double[][] matrix, double[] target) {
        int len = matrix.length;
        for (int i = 0; i < len; i++) {
            int index = i;
            for (int j = i; j < len; j++) if (Math.abs(matrix[j][i]) > Math.abs(matrix[index][i])) index = j;
            for (int j = 0; j < len; j++) {
                double tmp = matrix[i][j];
                matrix[i][j] = matrix[index][j];
                matrix[index][j] = tmp;
            }
            double tmp2 = target[i];
            target[i] = target[index];
            target[index] = tmp2;
            double mult = matrix[i][i];
            for (int j = i; j < len; j++) {
                matrix[i][j] /= mult;
            }
            target[i] /= mult;
            for (int j = i + 1; j < len; j++) {
                double mult2 = matrix[j][i];
                for (int k = i; k < len; k++) {
                    matrix[j][k] -= mult2 * matrix[i][k];
                }
                target[j] -= mult2 * target[i];
            }
            //            for (double[] d: copied) System.out.println(arrayToString(d));
            //            for (double[] d: res) System.out.println(arrayToString(d));
            //            System.out.println();
        }
        for (int i = len - 1; i > 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                double mult2 = matrix[j][i];
                target[j] -= mult2 * target[i];
            }
        }
        return target;
    }

    public static double[] calcPseudoReverse(String fileName) throws IOException {
        BufferedReader r = new BufferedReader(new FileReader(new File(fileName)));
        int m = nextInt(r);
        int n = nextInt(r);
        double[][] matrix = new double[n][m + 1];
        double[][] target = new double[n][1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                matrix[i][j] = nextDouble(r);
            }
            matrix[i][m] = 1;
            target[i][0] = nextDouble(r);
        }
        int testN = nextInt(r);
        double[][] testMatrix = new double[testN][m + 1];
        double[][] testTarget = new double[testN][1];
        for (int i = 0; i < testN; i++) {
            for (int j = 0; j < m; j++) {
                testMatrix[i][j] = nextDouble(r);
            }
            testMatrix[i][m] = 1;
            testTarget[i][0] = nextDouble(r);
        }
        double[] right = transpose(multiply(transpose(matrix), target))[0];
        return solveGauss(multiply(transpose(matrix), matrix), right);
    }

    public static void main(String[] args) throws IOException {
        for (int i = 1; i < 8; i++) {
            double[] res1 = calcGradientDescent("src/main/resources/" + i + ".txt", i);
            double[] res2 = calcPseudoReverse("src/main/resources/" + i + ".txt");
            System.out.println(i + "th set, gradient descent:");
            for (double d: res1) System.out.print(String.format("%.6f ", d));
            System.out.println();
            System.out.println(i + "th set, pseudo-reverse matrix:");
            for (double d: res2) System.out.print(String.format("%.6f ", d));
            System.out.println();
        }
    }

}