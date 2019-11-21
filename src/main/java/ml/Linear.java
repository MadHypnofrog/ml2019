package ml;

import javafx.util.Pair;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import static ml.Utils.nextInt;

public class Linear {

    private static Pair<double[], Double> calcGradientDescent(int num, double matrixQ,
                                                              int m, int n, double testN, double[][] matrix, double[] target,
                                                              double[][] testMatrix, double[] testTarget, double minY, double minYTest,
                                                              double maxY, double maxYTest) {
        Random R = new Random();
        double[] res = new double[m + 1];
        for (int i = 0; i < m + 1; i++) res[i] = R.nextDouble() / n;
        double[] grad = new double[m + 1];
        double[] pred = new double[n];
        double[] errors = new double[n];
        double[] xGrad = new double[n];
        double step = 10e-35;
        XYSeriesCollection ds = new XYSeriesCollection();
        XYSeriesCollection ds2 = new XYSeriesCollection();
        XYSeries series = new XYSeries("Gradient descent");
        XYSeries series2 = new XYSeries("Gradient descent");
        XYSeries series3 = new XYSeries("Pseudoinverse matrix");
        double qTest = 0;
        for (int iter = 0; iter < 3000; iter++) {
            for (int i = 0; i < n; i++) {
                pred[i] = Utils.scalarProduct(res, matrix[i]);
                errors[i] = target[i] - pred[i];
            }
            for (int i = 0; i < m + 1; i++) {
                grad[i] = 0;
                for (int k = 0; k < n; k++) {
                    grad[i] -= errors[k] * matrix[k][i];
                }
                grad[i] = grad[i] * 2 / n;
            }
            for (int i = 0; i < n; i++) xGrad[i] = Utils.scalarProduct(grad, matrix[i]);
            double sumLeft = 0;
            double sumRight = 0;
            for (int i = 0; i < n; i++) {
                sumLeft = sumLeft + errors[i] * xGrad[i];
                sumRight -= xGrad[i] * xGrad[i];
            }
            if (Math.abs(sumLeft / sumRight) < step) break;
            for (int i = 0; i < m + 1; i++) {
                res[i] -= sumLeft * grad[i] / sumRight;
            }
            double q = 0;
            for (int i = 0; i < n; i++) {
                q += (Utils.scalarProduct(matrix[i], res) - target[i]) * (Utils.scalarProduct(matrix[i], res) - target[i]);
            }
            q /= n;
            series.add(iter + 1, Math.sqrt(q) / (maxY - minY));
            qTest = 0;
            for (int i = 0; i < testN; i++) {
                qTest += (Utils.scalarProduct(testMatrix[i], res) - testTarget[i]) * (Utils.scalarProduct(testMatrix[i], res) - testTarget[i]);
            }
            qTest /= testN;
            qTest = Math.sqrt(qTest) / (maxYTest - minYTest);
            series2.add(iter + 1, qTest);
            series3.add(iter + 1, matrixQ);
        }
        ds.addSeries(series);
        ds2.addSeries(series2);
        ds2.addSeries(series3);
        Utils.writeChartForDS("Q(w) to Iterations (training set " + num + ")", ds, "Iterations", "Q(w)");
        Utils.writeChartForDS("Q(w) to Iterations (test set " + num + ")", ds2, "Iterations", "Q(w)");
        return new Pair<>(res, qTest);
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
            if (mult == 0) {
                for (int j = 0; j < len; j++) matrix[i][j] = 0;
                target[i] = 0;
                continue;
            }
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
        }
        for (int i = len - 1; i > 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                double mult2 = matrix[j][i];
                target[j] -= mult2 * target[i];
            }
        }
        return target;
    }

    private static Pair<double[], Double> calcPseudoReverse(double testN, double[][] matrix, double[] target,
                                                            double[][] testMatrix, double[] testTarget, double minYTest,
                                                            double maxYTest) {
        double[][] targetT = new double[target.length][1];
        for (int i = 0; i < target.length; i++) targetT[i][0] = target[i];
        double[] right = Utils.transpose(Utils.multiply(Utils.transpose(matrix), targetT))[0];
        double[] res = solveGauss(Utils.multiply(Utils.transpose(matrix), matrix), right);
        double qTest = 0;
        for (int i = 0; i < testN; i++) {
            qTest += (Utils.scalarProduct(testMatrix[i], res) - testTarget[i])
                    * (Utils.scalarProduct(testMatrix[i], res) - testTarget[i]);
        }
        qTest /= testN;
        return new Pair<>(res, Math.sqrt(qTest) / (maxYTest - minYTest));
    }

    public static void main(String[] args) throws IOException {
        for (int test = 1; test < 8; test++) {
            BufferedReader r = new BufferedReader(new FileReader(new File("src/main/resources/" + test + ".txt")));
            int m = (int)nextInt(r);
            int n = (int)nextInt(r);
            double[][] matrix = new double[n][m + 1];
            double[] target = new double[n];
            double maxY = -100000000000D;
            double minY = 100000000000D;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    matrix[i][j] = nextInt(r);
                }
                matrix[i][m] = 1;
                target[i] = nextInt(r);
                if (maxY < target[i]) maxY = target[i];
                if (minY > target[i]) minY = target[i];
            }
            int testN = (int)nextInt(r);
            double[][] testMatrix = new double[testN][m + 1];
            double[] testTarget = new double[testN];
            double maxYTest = -100000000000D;
            double minYTest = 100000000000D;
            for (int i = 0; i < testN; i++) {
                for (int j = 0; j < m; j++) {
                    testMatrix[i][j] = nextInt(r);
                }
                testMatrix[i][m] = 1;
                testTarget[i] = nextInt(r);
                if (maxYTest < testTarget[i]) maxYTest = testTarget[i];
                if (minYTest > testTarget[i]) minYTest = testTarget[i];
            }
            Pair<double[], Double> resMatrix = calcPseudoReverse(testN, matrix, target, testMatrix, testTarget,
                    minYTest, maxYTest);
            Pair<double[], Double> resDescent = calcGradientDescent(test, resMatrix.getValue(), m, n, testN, matrix, target, testMatrix,
                    testTarget, minY, minYTest, maxY, maxYTest);
            System.out.println(test + "th set, gradient descent:");
            for (double d: resDescent.getKey()) System.out.print(String.format("%.6f ", d));
            System.out.println();
            System.out.println("Q = " + resDescent.getValue());
            System.out.println(test + "th set, pseudo-reverse matrix:");
            for (double d: resMatrix.getKey()) System.out.print(String.format("%.6f ", d));
            System.out.println();
            System.out.println("Q = " + resMatrix.getValue());
        }
    }

}