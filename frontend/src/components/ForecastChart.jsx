import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const ForecastChart = ({ data, currency, color, horizon }) => {
  if (!data || !data.length) return null;

  // We expect data to be an array of objects: { ds, yhat, yhat_lower, yhat_upper }
  const labels = data.map(d => d.ds);

  const chartData = {
    labels,
    datasets: [
      {
        label: `${currency}/INR Forecast`,
        data: data.map(d => d.yhat),
        borderColor: color,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 1,
        tension: 0.4,
        fill: false,
        zIndex: 10
      },
      {
        label: 'Upper Bound (95%)',
        data: data.map(d => d.yhat_upper),
        borderColor: 'transparent',
        backgroundColor: `${color}20`, // e.g., 20% opacity of the theme color
        fill: '+1',                     // Fill down to the next dataset (lower bound)
        pointRadius: 0,
        tension: 0.4,
      },
      {
        label: 'Lower Bound (95%)',
        data: data.map(d => d.yhat_lower),
        borderColor: 'transparent',
        backgroundColor: 'transparent',
        fill: false,
        pointRadius: 0,
        tension: 0.4,
      }
    ]
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const label = context.dataset.label || '';
            if (label.includes('Bound')) return null; // hide bounds from tooltip to avoid clutter
            const val = context.parsed.y.toFixed(4);
            const dataIndex = context.dataIndex;
            const upper = data[dataIndex].yhat_upper.toFixed(4);
            const lower = data[dataIndex].yhat_lower.toFixed(4);
            return [
              `Forecast: ₹${val}`,
              `95% Range: ₹${lower} - ₹${upper}`
            ];
          }
        }
      }
    },
    scales: {
      y: {
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#888' }
      },
      x: {
        grid: { display: false },
        ticks: { color: '#bbb', maxTicksLimit: 8 }
      }
    }
  };

  return (
    <div style={{ height: '220px', width: '100%', marginTop: '16px' }}>
      <h4 style={{ fontSize: '0.8rem', color: '#888', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>
        {horizon}-Day Trajectory & Confidence Band
      </h4>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default ForecastChart;
