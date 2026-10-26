<h1 align="center">Leaderboard</h1>

<p align="center">Omni-InteractScope results across 12 model configurations and eight task groups.</p>

<p align="center"><a href="docs/protocol.md">Evaluation protocol</a> · <a href="docs/models.md">Run an evaluation</a> · <a href="docs/public_results.json">Result counts</a></p>

## Evaluation on explicit-request tasks (T1–T7)

Joint content–timing accuracy (%), higher is better. **Avg.** equally weights T1–T4, T6 and T7.

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">Size</th>
      <th align="center">Input</th>
      <th align="center">T1</th>
      <th align="center">T2</th>
      <th align="center">T3</th>
      <th align="center">T4</th>
      <th align="center">T5<sup>†</sup></th>
      <th align="center">T6</th>
      <th align="center">T7</th>
      <th align="center"><strong>Avg.</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr><th colspan="11" align="left">Proprietary models</th></tr>
    <tr>
      <td align="left">Gemini-3.1-Pro</td>
      <td align="center">N/R</td>
      <td align="center">V, A</td>
      <td align="right">22.5</td>
      <td align="right">28.2</td>
      <td align="right">11.6</td>
      <td align="right"><strong>11.2</strong></td>
      <td align="right">N/A</td>
      <td align="right"><strong>27.0</strong></td>
      <td align="right">11.2</td>
      <td align="right">18.6</td>
    </tr>
    <tr>
      <td align="left">Gemini-3.5-Flash</td>
      <td align="center">N/R</td>
      <td align="center">V, A</td>
      <td align="right">29.5</td>
      <td align="right"><strong>31.6</strong></td>
      <td align="right"><strong>14.0</strong></td>
      <td align="right">10.2</td>
      <td align="right">N/A</td>
      <td align="right">24.3</td>
      <td align="right"><strong>11.4</strong></td>
      <td align="right"><strong>20.2</strong></td>
    </tr>
    <tr>
      <td align="left">Qwen3.5-Omni-Plus</td>
      <td align="center">N/R</td>
      <td align="center">V, A</td>
      <td align="right"><strong>32.0</strong></td>
      <td align="right">24.3</td>
      <td align="right">4.8</td>
      <td align="right">3.9</td>
      <td align="right">N/A</td>
      <td align="right">10.8</td>
      <td align="right">7.9</td>
      <td align="right">13.9</td>
    </tr>
    <tr><th colspan="11" align="left">Open-source models</th></tr>
    <tr>
      <td align="left">InternVL3.5</td>
      <td align="center">8B</td>
      <td align="center">V+ASR</td>
      <td align="right">29.0</td>
      <td align="right">1.0</td>
      <td align="right">0.0</td>
      <td align="right">0.0</td>
      <td align="right">N/A</td>
      <td align="right">5.4</td>
      <td align="right">0.4</td>
      <td align="right">6.0</td>
    </tr>
    <tr>
      <td align="left">Qwen3-VL</td>
      <td align="center">8B</td>
      <td align="center">V+ASR</td>
      <td align="right">19.5</td>
      <td align="right">4.9</td>
      <td align="right">0.5</td>
      <td align="right">0.0</td>
      <td align="right">N/A</td>
      <td align="right">2.7</td>
      <td align="right">3.1</td>
      <td align="right">5.1</td>
    </tr>
    <tr>
      <td align="left">Qwen2.5-Omni</td>
      <td align="center">7B</td>
      <td align="center">V, A</td>
      <td align="right">11.0</td>
      <td align="right">0.5</td>
      <td align="right">0.5</td>
      <td align="right">0.5</td>
      <td align="right">N/A</td>
      <td align="right">0.0</td>
      <td align="right">1.5</td>
      <td align="right">2.3</td>
    </tr>
    <tr>
      <td align="left">Qwen3-Omni</td>
      <td align="center">30B-A3B</td>
      <td align="center">V, A</td>
      <td align="right">26.5</td>
      <td align="right">2.9</td>
      <td align="right">1.4</td>
      <td align="right">1.0</td>
      <td align="right">N/A</td>
      <td align="right">2.7</td>
      <td align="right">5.7</td>
      <td align="right">6.7</td>
    </tr>
    <tr><th colspan="11" align="left">Streaming Models</th></tr>
    <tr>
      <td align="left">LiveCC</td>
      <td align="center">7B</td>
      <td align="center">V</td>
      <td align="right">0.0</td>
      <td align="right">1.5</td>
      <td align="right">0.5</td>
      <td align="right">0.0</td>
      <td align="right">0.0</td>
      <td align="right">0.0</td>
      <td align="right">0.7</td>
      <td align="right">0.4</td>
    </tr>
    <tr>
      <td align="left">AURA</td>
      <td align="center">8B</td>
      <td align="center">V</td>
      <td align="right">0.0</td>
      <td align="right">4.9</td>
      <td align="right">0.5</td>
      <td align="right">1.5</td>
      <td align="right">0.0</td>
      <td align="right">2.7</td>
      <td align="right">0.7</td>
      <td align="right">1.7</td>
    </tr>
    <tr>
      <td align="left">MMDuet2</td>
      <td align="center">3B</td>
      <td align="center">V</td>
      <td align="right">0.0</td>
      <td align="right">3.4</td>
      <td align="right">1.0</td>
      <td align="right">0.5</td>
      <td align="right">0.0</td>
      <td align="right">2.7</td>
      <td align="right">1.8</td>
      <td align="right">1.6</td>
    </tr>
    <tr>
      <td align="left">MOSS-VL</td>
      <td align="center">11.3B</td>
      <td align="center">V</td>
      <td align="right">1.5</td>
      <td align="right">9.2</td>
      <td align="right">1.9</td>
      <td align="right">5.8</td>
      <td align="right">0.9</td>
      <td align="right">8.1</td>
      <td align="right">5.9</td>
      <td align="right">5.4</td>
    </tr>
    <tr>
      <td align="left">MiniCPM-o 4.5</td>
      <td align="center">9B</td>
      <td align="center">V, A</td>
      <td align="right">1.5</td>
      <td align="right">8.3</td>
      <td align="right">1.4</td>
      <td align="right">1.0</td>
      <td align="right">1.7</td>
      <td align="right">2.7</td>
      <td align="right">1.1</td>
      <td align="right">2.7</td>
    </tr>
  </tbody>
</table>

T1: tool invocation; T2: event notification; T3: risk/error interjection; T4: cumulative-state
tracking; T5: temporal conditions; T6: nonverbal/workflow cues; T7: continuous responses.
There are 200/206/207/206/115/37/457 targets for T1–T7, respectively.

† T5 uses the earlier query-relative rolling evaluation and is reported separately, excluded
from every Avg. N/A denotes prompted inputs without an explicit elapsed-time clock.
T1–T4, T6 and T7 use 1,063 interactions and 1,313 targets.

## Evaluation on assistance tasks (T8)

Timely reference-target recall (%), higher is better. Overall pools hits over all 360 targets.
T8-A is query-free reminders (122 targets), T8-B audiovisual QA (151), and T8-C guidance (87).
Complete T8-C counts fully covered guidance interactions out of 26.

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">Size</th>
      <th align="center">Input</th>
      <th align="center"><strong>Overall</strong></th>
      <th align="center">T8-A</th>
      <th align="center">T8-B</th>
      <th align="center">T8-C</th>
      <th align="center">Complete T8-C</th>
    </tr>
  </thead>
  <tbody>
    <tr><th colspan="8" align="left">Proprietary models</th></tr>
    <tr>
      <td align="left">Gemini-3.1-Pro</td>
      <td align="center">N/R</td>
      <td align="center">V, A</td>
      <td align="right"><strong>46.39</strong></td>
      <td align="right"><strong>22.95</strong></td>
      <td align="right"><strong>62.25</strong></td>
      <td align="right"><strong>51.72</strong></td>
      <td align="right"><strong>3/26</strong></td>
    </tr>
    <tr>
      <td align="left">Gemini-3.5-Flash</td>
      <td align="center">N/R</td>
      <td align="center">V, A</td>
      <td align="right">41.11</td>
      <td align="right">22.13</td>
      <td align="right">57.62</td>
      <td align="right">39.08</td>
      <td align="right">2/26</td>
    </tr>
    <tr>
      <td align="left">Qwen3.5-Omni-Plus</td>
      <td align="center">N/R</td>
      <td align="center">V, A</td>
      <td align="right">31.11</td>
      <td align="right">9.02</td>
      <td align="right">49.67</td>
      <td align="right">29.89</td>
      <td align="right">1/26</td>
    </tr>
    <tr><th colspan="8" align="left">Open-source models</th></tr>
    <tr>
      <td align="left">InternVL3.5</td>
      <td align="center">8B</td>
      <td align="center">V+ASR</td>
      <td align="right">14.17</td>
      <td align="right">0.00</td>
      <td align="right">26.49</td>
      <td align="right">12.64</td>
      <td align="right">0/26</td>
    </tr>
    <tr>
      <td align="left">Qwen3-VL</td>
      <td align="center">8B</td>
      <td align="center">V+ASR</td>
      <td align="right">15.00</td>
      <td align="right">2.46</td>
      <td align="right">24.50</td>
      <td align="right">16.09</td>
      <td align="right">0/26</td>
    </tr>
    <tr>
      <td align="left">Qwen2.5-Omni</td>
      <td align="center">7B</td>
      <td align="center">V, A</td>
      <td align="right">10.00</td>
      <td align="right">2.46</td>
      <td align="right">19.87</td>
      <td align="right">3.45</td>
      <td align="right">0/26</td>
    </tr>
    <tr>
      <td align="left">Qwen3-Omni</td>
      <td align="center">30B-A3B</td>
      <td align="center">V, A</td>
      <td align="right">13.06</td>
      <td align="right">2.46</td>
      <td align="right">25.83</td>
      <td align="right">5.75</td>
      <td align="right">0/26</td>
    </tr>
    <tr><th colspan="8" align="left">Streaming Models</th></tr>
    <tr>
      <td align="left">LiveCC</td>
      <td align="center">7B</td>
      <td align="center">V</td>
      <td align="right">3.61</td>
      <td align="right">0.00</td>
      <td align="right">8.61</td>
      <td align="right">0.00</td>
      <td align="right">0/26</td>
    </tr>
    <tr>
      <td align="left">AURA</td>
      <td align="center">8B</td>
      <td align="center">V</td>
      <td align="right">6.94</td>
      <td align="right">0.00</td>
      <td align="right">15.89</td>
      <td align="right">1.15</td>
      <td align="right">0/26</td>
    </tr>
    <tr>
      <td align="left">MMDuet2</td>
      <td align="center">3B</td>
      <td align="center">V</td>
      <td align="right">1.94</td>
      <td align="right">0.00</td>
      <td align="right">4.64</td>
      <td align="right">0.00</td>
      <td align="right">0/26</td>
    </tr>
    <tr>
      <td align="left">MOSS-VL</td>
      <td align="center">11.3B</td>
      <td align="center">V</td>
      <td align="right">3.33</td>
      <td align="right">0.00</td>
      <td align="right">7.28</td>
      <td align="right">1.15</td>
      <td align="right">0/26</td>
    </tr>
    <tr>
      <td align="left">MiniCPM-o 4.5</td>
      <td align="center">9B</td>
      <td align="center">V, A</td>
      <td align="right">3.06</td>
      <td align="right">0.00</td>
      <td align="right">7.28</td>
      <td align="right">0.00</td>
      <td align="right">0/26</td>
    </tr>
  </tbody>
</table>

V: video; A: native audio; ASR: scene transcripts; N/R: undisclosed model size;
A3B: active parameters. The first seven models use independent prompted calls; the final
five use rolling evaluation. Unmatched T8 responses are not penalized by recall.
See [scoring protocol](docs/protocol.md) for response timing and matching rules.
