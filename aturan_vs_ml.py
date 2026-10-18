"""
Studi Kasus: Aturan Manual vs Machine Learning + Audit Bias (persetujuan pinjaman)
Berdasarkan kuliah umum "Introduction to Machine Learning" (Dr. Taniza Tajudin, UiTM Kedah).

Poin materi yang dipraktikkan:
  - Pemrograman tradisional (aturan ditulis programmer) vs ML (pola dipelajari dari data)
  - Supervised learning: classification + evaluasi akurasi
  - Etika ML: bias, transparansi
Data SINTETIS. Bias historis sengaja ditanam untuk demonstrasi; tidak merepresentasikan lembaga mana pun.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DOCS = Path("docs")
rng = np.random.default_rng(2026)
sigmoid = lambda z: 1 / (1 + np.exp(-z))


def buat_data(n=6000):
    desa = (rng.random(n) < 0.4).astype(int)                       # 1 = Pedesaan, 0 = Perkotaan
    income = rng.lognormal(np.log(6), 0.5, n) * np.where(desa, 0.85, 1)   # juta rupiah / bulan
    score = np.clip(rng.normal(650, 70, n) - 12 * desa, 300, 850)         # skor kredit
    debt = np.clip(rng.normal(0.32, 0.12, n), 0.02, 0.8)                  # rasio utang
    jarak = np.where(desa, rng.gamma(4, 6, n), rng.gamma(2, 2.5, n))      # km ke cabang (proksi wilayah)
    # kemampuan bayar sebenarnya (kebenaran dasar)
    z = 1.2 * np.log(income / 6) + 0.012 * (score - 650) - 4 * (debt - 0.32) + 0.8
    layak = (rng.random(n) < sigmoid(z)).astype(int)
    # keputusan HISTORIS: ikut kemampuan bayar, tetapi bias terhadap pedesaan
    hist = (rng.random(n) < sigmoid(1.3 * (z - 1.0 * desa + rng.normal(0, .4, n)))).astype(int)
    return pd.DataFrame(dict(income=income, score=score, debt=debt, desa=desa, jarak=jarak, layak=layak, disetujui=hist))


def aturan(df, ti, ts, td):
    return ((df.income >= ti) & (df.score >= ts) & (df.debt <= td)).astype(int)


def cari_aturan_terbaik(train):
    best = (0, None)
    for ti in np.arange(2, 12.1, .5):
        for ts in range(450, 751, 10):
            for td in np.arange(.1, .71, .05):
                acc = (aturan(train, ti, ts, td) == train.disetujui).mean()
                if acc > best[0]:
                    best = (acc, (float(ti), int(ts), round(float(td), 2)))
    return best


def main():
    df = buat_data()
    train, test = df.iloc[:4800].copy(), df.iloc[4800:].copy()
    acc, (ti, ts, td) = cari_aturan_terbaik(train)
    print(f"Aturan terbaik: income>={ti} jt, skor>={ts}, utang<={td} (akurasi latih {acc:.1%})")

    varian = {"m0": ["income", "score", "debt", "desa", "jarak"],
              "m1": ["income", "score", "debt", "jarak"],      # tanpa atribut wilayah, proksi tetap ada
              "m2": ["income", "score", "debt"]}               # tanpa wilayah & proksi
    coef = None
    for k, cols in varian.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[cols], train.disetujui)
        test[k] = model.predict(test[cols])
        if k == "m0":
            coef = sorted(({"f": c, "c": round(float(v), 2)} for c, v in zip(cols, model[-1].coef_[0])),
                          key=lambda r: -abs(r["c"]))

    rows = [{"i": round(r.income, 2), "s": int(r.score), "d": round(r.debt, 2), "w": int(r.desa), "o": int(r.disetujui),
             "g": int(r.layak), "m0": int(r.m0), "m1": int(r.m1), "m2": int(r.m2)} for r in test.itertuples()]
    meta = {"rule": {"i": ti, "s": ts, "d": td}, "coef": coef, "n_train": len(train), "n_test": len(test)}

    # ringkasan cetak (juga untuk laporan)
    for nama, pred in [("Aturan manual", aturan(test, ti, ts, td))] + [(k, test[k]) for k in varian]:
        a = (pred == test.disetujui).mean()
        tpr = [pred[(test.layak == 1) & (test.desa == w)].mean() for w in (0, 1)]
        print(f"{nama:14} akurasi {a:.1%} | peluang disetujui (yang layak): kota {tpr[0]:.1%}, desa {tpr[1]:.1%}, selisih {100*(tpr[0]-tpr[1]):.1f} poin")

    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").touch()
    test.to_csv(DOCS / "data_uji_dan_prediksi.csv", index=False)
    html = TEMPLATE.replace("__DATA__", json.dumps(rows, separators=(",", ":"))).replace("__META__", json.dumps(meta))
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    print("Selesai -> docs/index.html")


TEMPLATE = r'''<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Aturan vs Machine Learning</title>
<style>
:root{--ink:#111;--bg:#fffbee;--y:#ffd23f;--p:#ff9ecd;--g:#7ee8a2;--b:#7cc4ff;--r:#ff6b6b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:"Space Grotesk","Helvetica Neue",Arial,sans-serif;line-height:1.45}
.wrap{max-width:1240px;margin:auto;padding:28px 22px 44px}
h1{margin:0;font-size:clamp(2rem,6vw,3.5rem);font-weight:800;letter-spacing:-.03em;line-height:1.05}
h1 mark{background:var(--y);padding:0 .15em;border:3px solid var(--ink);box-shadow:5px 5px 0 var(--ink);display:inline-block;transform:rotate(-1.5deg)}
.sub{margin:14px 0 20px;font-weight:600;max-width:780px}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;background:#fff;border:3px solid var(--ink);box-shadow:8px 8px 0 var(--ink);padding:12px 14px;margin-bottom:26px;font-weight:700;font-size:.88rem}
.pill{border:3px solid var(--ink);padding:5px 12px;background:var(--b)}.pill.y{background:var(--y)}.pill.r{background:var(--r)}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:22px;align-items:start}.s5{grid-column:span 5}.s7{grid-column:span 7}.s6{grid-column:span 6}.s12{grid-column:span 12}
.card{background:#fff;border:3px solid var(--ink);box-shadow:7px 7px 0 var(--ink);padding:16px 18px}.card h2{margin:0 0 12px;font-size:1rem;text-transform:uppercase}
.sl{margin-bottom:14px}.sl label{display:flex;justify-content:space-between;font-weight:700;font-size:.85rem}.sl input{width:100%;accent-color:#111;margin:6px 0 0}
.kp{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:6px}.kp div{border:3px solid var(--ink);box-shadow:4px 4px 0 var(--ink);padding:8px 10px}
.kp span{display:block;font-size:.7rem;font-weight:700;text-transform:uppercase}.kp b{font-size:1.5rem}
table{border-collapse:collapse;width:100%;font-size:.84rem;border:3px solid var(--ink)}th{background:var(--ink);color:#fff;padding:8px;font-size:.7rem;text-transform:uppercase;text-align:left}
td{padding:8px;border-top:2px solid var(--ink);font-weight:600}tr.live td:first-child{background:var(--y)}td.bad{background:var(--r)}td.mid{background:var(--y)}td.ok{background:var(--g)}
.gb{display:flex;align-items:center;gap:8px;margin:7px 0;font-size:.82rem;font-weight:600}.gb span{min-width:190px}.gb i{display:block;height:20px;border:2px solid var(--ink);min-width:3px}
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}.how div{border:3px solid var(--ink);padding:10px 12px;font-size:.88rem;font-weight:500}.how b{display:block;text-transform:uppercase;font-size:.78rem;margin-bottom:3px}
.note{margin:14px 0 0;border:3px solid var(--ink);background:var(--r);padding:8px 12px;font-weight:700;font-size:.85rem}.ins{border:3px solid var(--ink);background:var(--g);padding:10px 12px;font-weight:700;margin-top:14px;font-size:.9rem}
@media(max-width:860px){.s5,.s7,.s6{grid-column:span 12}.gb span{min-width:120px}}
</style></head><body><div class="wrap">
<h1>Aturan <mark>vs</mark> Machine Learning</h1>
<p class="sub">Persetujuan pinjaman diputuskan dua cara: aturan <code>if-else</code> buatan manusia dan model ML yang belajar dari data historis. Geser ambang aturan, lalu lihat akurasi dan keadilannya antarwilayah.</p>
<div class="bar"><span class="pill y">Data sintetis</span><span class="pill" id="pn"></span><span>Studi kasus kuliah umum Introduction to Machine Learning (Dr. Taniza Tajudin, UiTM)</span></div>
<div class="grid">
<div class="card s5"><h2>Aturan manual (geser ambangnya)</h2>
<div class="sl"><label>Penghasilan minimum <b id="vi"></b></label><input type="range" id="ri" min="2" max="12" step="0.5"></div>
<div class="sl"><label>Skor kredit minimum <b id="vs"></b></label><input type="range" id="rs" min="450" max="750" step="10"></div>
<div class="sl"><label>Rasio utang maksimum <b id="vd"></b></label><input type="range" id="rd" min="0.1" max="0.7" step="0.05"></div>
<div class="kp"><div style="background:var(--b)"><span>Akurasi</span><b id="k1"></b></div><div style="background:var(--p)"><span>Selisih peluang disetujui</span><b id="k2"></b></div>
<div style="background:var(--g)"><span>Layak disetujui · Kota</span><b id="k3"></b></div><div style="background:var(--y)"><span>Layak disetujui · Desa</span><b id="k4"></b></div></div></div>
<div class="card s7"><h2>Perbandingan metode</h2><div style="overflow-x:auto"><table id="tb"></table></div>
<div id="gaps" style="margin-top:14px"></div><div class="ins" id="ins"></div></div>
<div class="card s6"><h2>Apa yang dipelajari model ML penuh?</h2><div id="coef"></div>
<p style="font-size:.85rem;font-weight:500;margin:10px 0 0">Koefisien model regresi logistik (nilai terstandarisasi). Tanda negatif pada <b>desa</b> berarti model belajar menolak pemohon pedesaan, karena begitulah data historisnya.</p></div>
<div class="card s6"><h2>Cara membaca</h2><div class="how">
<div style="background:var(--b)"><b>Akurasi</b>Seberapa sering keputusan metode sama dengan keputusan historis pada data uji.</div>
<div style="background:var(--g)"><b>Peluang disetujui</b>Dari pemohon yang sebenarnya layak bayar, berapa persen yang disetujui. Idealnya sama di kota dan desa.</div>
<div style="background:var(--p)"><b>Selisih</b>Kota dikurangi desa, dalam poin persen. Makin besar, makin tidak adil.</div>
<div style="background:var(--y)"><b>Tanpa wilayah</b>Menghapus kolom wilayah belum tentu cukup: jarak ke cabang bisa menjadi proksi.</div></div></div>
<div class="card s12"><h2>Pelajaran</h2><div class="how">
<div><b>Aturan = transparan</b>Setiap keputusan bisa dijelaskan, tetapi ambangnya dipilih manusia dan bisa tetap bias.</div>
<div><b>ML = belajar dari data</b>Pola ditemukan otomatis, tetapi bias dalam data ikut dipelajari.</div>
<div><b>Akurasi tinggi ≠ adil</b>Model bisa mirip keputusan historis sekaligus mengulang ketidakadilannya.</div>
<div><b>Audit itu wajib</b>Bandingkan hasil antarkelompok sebelum model dipakai (materi: bias, privasi, transparansi).</div></div>
<p class="note">Data dan bias dalam halaman ini sengaja dibuat untuk pembelajaran (sintetis). Tidak merepresentasikan lembaga keuangan mana pun, dan bukan untuk dipakai menilai pemohon sebenarnya.</p></div>
</div></div>
<script>
const T=__DATA__,M=__META__,$=i=>document.getElementById(i),pct=x=>isNaN(x)?'-':(x*100).toFixed(1)+'%';
$('pn').textContent=T.length+' pemohon uji';
function met(f){let ok=0,a=[0,0],n=[0,0],g=[0,0],gt=[0,0];
 T.forEach(r=>{const p=f(r);ok+=p===r.o;n[r.w]++;a[r.w]+=p;if(r.g){g[r.w]++;gt[r.w]+=p}});
 return{acc:ok/T.length,ar:[a[0]/n[0],a[1]/n[1]],tpr:[gt[0]/g[0],gt[1]/g[1]]}}
let ti=M.rule.i,ts=M.rule.s,td=M.rule.d;
const rule=r=>(r.i>=ti&&r.s>=ts&&r.d<=td)?1:0;
const METH=[{n:'Aturan manual (live)',f:rule,live:1},{n:'ML penuh (pakai wilayah)',f:r=>r.m0},{n:'ML tanpa wilayah',f:r=>r.m1},{n:'ML tanpa wilayah & proksi',f:r=>r.m2}];
function gapCls(g){return g>=8?'bad':g>=4?'mid':'ok'}
function render(){
 $('vi').textContent=ti+' jt';$('vs').textContent=ts;$('vd').textContent=td;
 const R=METH.map(m=>({...m,m:met(m.f)})).map(x=>({...x,gap:(x.m.tpr[0]-x.m.tpr[1])*100}));
 const a=R[0];$('k1').textContent=pct(a.m.acc);$('k2').textContent=a.gap.toFixed(1)+' poin';$('k3').textContent=pct(a.m.tpr[0]);$('k4').textContent=pct(a.m.tpr[1]);
 $('tb').innerHTML='<tr><th>Metode</th><th>Akurasi</th><th>Disetujui kota</th><th>Disetujui desa</th><th>Layak·kota</th><th>Layak·desa</th><th>Selisih</th></tr>'+
 R.map(x=>`<tr class="${x.live?'live':''}"><td>${x.n}</td><td>${pct(x.m.acc)}</td><td>${pct(x.m.ar[0])}</td><td>${pct(x.m.ar[1])}</td><td>${pct(x.m.tpr[0])}</td><td>${pct(x.m.tpr[1])}</td><td class="${gapCls(x.gap)}">${x.gap.toFixed(1)} poin</td></tr>`).join('');
 const cl={bad:'#ff6b6b',mid:'#ffd23f',ok:'#7ee8a2'};
 $('gaps').innerHTML='<b style="font-size:.8rem;text-transform:uppercase">Selisih peluang disetujui (kota − desa)</b>'+R.map(x=>`<div class="gb"><span>${x.n}</span><i style="width:${Math.min(60,Math.max(0,x.gap)*1.4)}%;background:${cl[gapCls(x.gap)]}"></i>${x.gap.toFixed(1)}</div>`).join('');
 const w=R.slice(1).reduce((p,c)=>c.gap>p.gap?c:p),b=R.slice(1).reduce((p,c)=>c.gap<p.gap?c:p);
 $('ins').textContent=`Di antara model ML, selisih terbesar ada pada "${w.n}" (${w.gap.toFixed(1)} poin) dan terkecil pada "${b.n}" (${b.gap.toFixed(1)} poin). Aturan manual saat ini: ${a.gap.toFixed(1)} poin dengan akurasi ${pct(a.m.acc)}.`;
}
[['ri',v=>ti=v],['rs',v=>ts=v],['rd',v=>td=v]].forEach(([id,fn])=>{const e=$(id);e.value={ri:ti,rs:ts,rd:td}[id];e.oninput=()=>{fn(+e.value);render()}});
const mx=Math.max(...M.coef.map(c=>Math.abs(c.c))),NM={income:'penghasilan',score:'skor kredit',debt:'rasio utang',desa:'wilayah desa',jarak:'jarak ke cabang'};
$('coef').innerHTML=M.coef.map(c=>`<div class="gb"><span>${NM[c.f]}</span><i style="width:${Math.abs(c.c)/mx*50}%;background:${c.c<0?'#ff6b6b':'#7ee8a2'}"></i>${c.c}</div>`).join('');
render();
</script></body></html>'''

if __name__ == "__main__":
    main()
