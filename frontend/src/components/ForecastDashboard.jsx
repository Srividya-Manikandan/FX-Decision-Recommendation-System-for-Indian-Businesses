import React, { useState, useEffect } from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { TrendingUp, TrendingDown, Activity, AlertTriangle, Info } from 'lucide-react';
import GlassCard from './GlassCard';
import axios from 'axios';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Filler,
    Legend
);

const API_BASE = 'http://localhost:5000/api';

const CURRENCY_COLORS = {
    USD: '#2563EB',
    GBP: '#10B981',
    EUR: '#6366F1',
    JPY: '#F59E0B',
};

// Helper: Format date string for display
const formatDate = (dateStr) => {
    const d = new Date(dateStr);
    const day = d.getDate();
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${day} ${months[d.getMonth()]}`;
};

const formatDateFull = (dateStr) => {
    const d = new Date(dateStr);
    const day = d.getDate();
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return `${day} ${months[d.getMonth()]} ${d.getFullYear()}`;
};

// ---------------------------------------------------------------------------
// Sub-Component: Single Currency Forecast Chart (Line + Confidence Band)
// ---------------------------------------------------------------------------
const ForecastLineChart = ({ forecastData, currency }) => {
    if (!forecastData || forecastData.status !== 'success' || !forecastData.forecast_table) {
        return (
            <div style={{ height: '280px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
                <Activity size={20} style={{ marginRight: '8px', opacity: 0.4 }} />
                Loading forecast...
            </div>
        );
    }

    const color = CURRENCY_COLORS[currency] || '#2563EB';
    const table = forecastData.forecast_table;

    const chartData = {
        labels: table.map(row => formatDate(row.ds)),
        datasets: [
            {
                label: 'Upper Bound (95%)',
                data: table.map(row => row.yhat_upper),
                borderColor: 'transparent',
                backgroundColor: `${color}20`,
                fill: '+1',
                pointRadius: 0,
                tension: 0.3,
                order: 2,
            },
            {
                label: 'Predicted Rate',
                data: table.map(row => row.yhat),
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: 3,
                pointRadius: 4,
                pointBackgroundColor: color,
                pointBorderColor: '#0A0A0B',
                pointBorderWidth: 2,
                tension: 0.3,
                fill: false,
                order: 1,
            },
            {
                label: 'Lower Bound (95%)',
                data: table.map(row => row.yhat_lower),
                borderColor: 'transparent',
                backgroundColor: `${color}20`,
                fill: '-1',
                pointRadius: 0,
                tension: 0.3,
                order: 3,
            },
        ],
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    color: '#9CA3AF',
                    font: { family: 'Inter', size: 11 },
                    usePointStyle: true,
                    pointStyle: 'circle',
                    filter: (item) => item.text === 'Predicted Rate' || item.text === 'Upper Bound (95%)',
                },
            },
            tooltip: {
                backgroundColor: '#16161a',
                titleColor: '#fff',
                bodyColor: '#9CA3AF',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                callbacks: {
                    title: (items) => {
                        const idx = items[0]?.dataIndex;
                        return idx !== undefined ? formatDateFull(table[idx].ds) : '';
                    },
                    label: (ctx) => `${ctx.dataset.label}: ₹${ctx.parsed.y.toFixed(4)}`,
                },
            },
        },
        scales: {
            y: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: '#9CA3AF', callback: (v) => `₹${v.toFixed(2)}` },
            },
            x: {
                grid: { display: false },
                ticks: { color: '#9CA3AF', font: { size: 10 } },
            },
        },
    };

    return (
        <div style={{ height: '280px' }}>
            <Line data={chartData} options={options} />
        </div>
    );
};

// ---------------------------------------------------------------------------
// Sub-Component: KPI Cards Row
// ---------------------------------------------------------------------------
const ForecastKPICards = ({ forecastData, currency }) => {
    if (!forecastData || forecastData.status !== 'success') return null;

    const color = CURRENCY_COLORS[currency] || '#2563EB';
    const trendUp = forecastData.trend === 'UP';

    const cards = [
        { label: 'Current Rate', value: `₹${forecastData.current_rate}`, color: '#fff' },
        { label: '7D Predicted', value: `₹${forecastData.predicted_rate}`, color: trendUp ? '#EF4444' : '#10B981' },
        { label: 'Upper Bound', value: `₹${forecastData.forecast_upper}`, color: '#F59E0B' },
        { label: 'Lower Bound', value: `₹${forecastData.forecast_lower}`, color: '#3B82F6' },
        {
            label: 'Trend',
            value: `${trendUp ? '↑' : '↓'} ${forecastData.change_percent > 0 ? '+' : ''}${forecastData.change_percent}%`,
            color: trendUp ? '#EF4444' : '#10B981',
        },
    ];

    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.8rem' }}>
            {cards.map((card, i) => (
                <div
                    key={i}
                    style={{
                        background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.06)',
                        borderRadius: '12px',
                        padding: '0.8rem 1rem',
                        textAlign: 'center',
                    }}
                >
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)', fontWeight: 600, letterSpacing: '0.5px', textTransform: 'uppercase', marginBottom: '6px' }}>
                        {card.label}
                    </div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 800, color: card.color, fontFamily: 'monospace' }}>
                        {card.value}
                    </div>
                </div>
            ))}
        </div>
    );
};

// ---------------------------------------------------------------------------
// Sub-Component: Message Banner
// ---------------------------------------------------------------------------
const ForecastMessageBanner = ({ forecastData }) => {
    if (!forecastData || forecastData.status !== 'success' || !forecastData.message) return null;

    return (
        <div style={{
            background: 'linear-gradient(135deg, rgba(37,99,235,0.08), rgba(16,185,129,0.05))',
            border: '1px solid rgba(37,99,235,0.15)',
            borderRadius: '12px',
            padding: '1rem 1.2rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.8rem',
        }}>
            <Info size={18} color="#2563EB" style={{ marginTop: '2px', flexShrink: 0 }} />
            <p style={{ color: '#D1D5DB', fontSize: '0.85rem', lineHeight: 1.6, margin: 0 }}>
                {forecastData.message}
            </p>
        </div>
    );
};

// ---------------------------------------------------------------------------
// Sub-Component: Multi-Currency Comparison Table
// ---------------------------------------------------------------------------
const ForecastComparisonTable = ({ allForecasts }) => {
    if (!allForecasts) return null;

    const currencies = ['USD', 'GBP', 'EUR', 'JPY'];

    return (
        <div style={{ overflowX: 'auto' }}>
            <table style={{
                width: '100%',
                borderCollapse: 'separate',
                borderSpacing: '0 4px',
            }}>
                <thead>
                    <tr>
                        {['Currency', 'Current', 'Predicted (7D)', 'Upper (95%)', 'Lower (95%)', 'Trend', 'Change %'].map(h => (
                            <th key={h} style={{
                                padding: '0.6rem 1rem',
                                textAlign: 'left',
                                fontSize: '0.65rem',
                                fontWeight: 700,
                                color: 'var(--text-secondary)',
                                textTransform: 'uppercase',
                                letterSpacing: '1px',
                                borderBottom: '1px solid rgba(255,255,255,0.06)',
                            }}>{h}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {currencies.map(cur => {
                        const d = allForecasts[cur];
                        if (!d || d.status !== 'success') {
                            return (
                                <tr key={cur}>
                                    <td style={{ padding: '0.7rem 1rem', color: CURRENCY_COLORS[cur], fontWeight: 700 }}>{cur}/INR</td>
                                    <td colSpan={6} style={{ padding: '0.7rem', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                                        <AlertTriangle size={14} style={{ marginRight: '6px' }} /> Forecast unavailable
                                    </td>
                                </tr>
                            );
                        }
                        const trendUp = d.trend === 'UP';
                        return (
                            <tr key={cur} style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '8px' }}>
                                <td style={{ padding: '0.7rem 1rem', fontWeight: 700, color: CURRENCY_COLORS[cur], fontSize: '0.85rem' }}>
                                    {cur}/INR
                                </td>
                                <td style={{ padding: '0.7rem 1rem', fontFamily: 'monospace', fontSize: '0.85rem' }}>₹{d.current_rate}</td>
                                <td style={{ padding: '0.7rem 1rem', fontFamily: 'monospace', fontSize: '0.85rem', fontWeight: 700, color: trendUp ? '#EF4444' : '#10B981' }}>
                                    ₹{d.predicted_rate}
                                </td>
                                <td style={{ padding: '0.7rem 1rem', fontFamily: 'monospace', fontSize: '0.8rem', color: '#F59E0B' }}>₹{d.forecast_upper}</td>
                                <td style={{ padding: '0.7rem 1rem', fontFamily: 'monospace', fontSize: '0.8rem', color: '#3B82F6' }}>₹{d.forecast_lower}</td>
                                <td style={{ padding: '0.7rem 1rem' }}>
                                    <span style={{
                                        display: 'inline-flex', alignItems: 'center', gap: '4px',
                                        padding: '2px 8px', borderRadius: '100px',
                                        background: trendUp ? 'rgba(239,68,68,0.12)' : 'rgba(16,185,129,0.12)',
                                        color: trendUp ? '#EF4444' : '#10B981',
                                        fontSize: '0.7rem', fontWeight: 800,
                                    }}>
                                        {trendUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                                        {d.trend}
                                    </span>
                                </td>
                                <td style={{
                                    padding: '0.7rem 1rem', fontFamily: 'monospace',
                                    fontWeight: 700, fontSize: '0.85rem',
                                    color: d.change_percent >= 0 ? '#EF4444' : '#10B981',
                                }}>
                                    {d.change_percent > 0 ? '+' : ''}{d.change_percent}%
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
};


// ---------------------------------------------------------------------------
// Main Export: ForecastDashboard
// ---------------------------------------------------------------------------
const ForecastDashboard = ({ horizon = 7, selectedDate = '' }) => {
    const [forecasts, setForecasts] = useState({});
    const [loading, setLoading] = useState(true);
    const [selectedCurrency, setSelectedCurrency] = useState('USD');

    useEffect(() => {
        const fetchAll = async () => {
            setLoading(true);
            try {
                const results = {};
                for (const cur of ['USD', 'GBP', 'EUR', 'JPY']) {
                    let url = `${API_BASE}/forecast?currency=${cur}&days=${horizon}`;
                    if (selectedDate) url += `&date=${selectedDate}`;
                    const res = await axios.get(url);
                    results[cur] = res.data;
                }
                setForecasts(results);
            } catch (err) {
                console.error('Forecast fetch error:', err);
            } finally {
                setLoading(false);
            }
        };
        fetchAll();
    }, [horizon, selectedDate]);

    if (loading) {
        return (
            <GlassCard title="Prophet Forecast Engine">
                <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    <Activity size={32} style={{ opacity: 0.3, marginBottom: '1rem' }} className="animate-spin" />
                    <p>Training Prophet models for all currencies...</p>
                    <p style={{ fontSize: '0.75rem', marginTop: '4px', opacity: 0.5 }}>This may take 15-30 seconds on first load.</p>
                </div>
            </GlassCard>
        );
    }

    const currentForecast = forecasts[selectedCurrency];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

            {/* Currency Selector Tabs */}
            <div style={{ display: 'flex', gap: '0.5rem' }}>
                {['USD', 'GBP', 'EUR', 'JPY'].map(cur => (
                    <button
                        key={cur}
                        onClick={() => setSelectedCurrency(cur)}
                        style={{
                            padding: '0.5rem 1.2rem',
                            borderRadius: '10px',
                            border: 'none',
                            background: selectedCurrency === cur ? CURRENCY_COLORS[cur] : 'rgba(255,255,255,0.05)',
                            color: selectedCurrency === cur ? '#fff' : 'var(--text-secondary)',
                            fontSize: '0.8rem',
                            fontWeight: 700,
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            letterSpacing: '0.5px',
                        }}
                    >
                        {cur}/INR
                    </button>
                ))}
            </div>

            {/* Component 2: KPI Cards */}
            <ForecastKPICards forecastData={currentForecast} currency={selectedCurrency} />

            {/* Component 1: Forecast Chart */}
            <GlassCard title={`${selectedCurrency}/INR — ${horizon}-Day Prophet Forecast`}>
                {currentForecast?.forecast_table && (
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginBottom: '0.5rem', letterSpacing: '0.5px' }}>
                        📅 {formatDateFull(currentForecast.forecast_table[0]?.ds)} → {formatDateFull(currentForecast.forecast_table[currentForecast.forecast_table.length - 1]?.ds)}
                    </p>
                )}
                <ForecastLineChart forecastData={currentForecast} currency={selectedCurrency} />
            </GlassCard>

            {/* Component 3: Message Banner */}
            <ForecastMessageBanner forecastData={currentForecast} />

            {/* Component 4: Multi-Currency Comparison */}
            <GlassCard title="Multi-Currency Forecast Comparison">
                <ForecastComparisonTable allForecasts={forecasts} />
            </GlassCard>
        </div>
    );
};

export default ForecastDashboard;
