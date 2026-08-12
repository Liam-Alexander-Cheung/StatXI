/* flags.js — the flag catalog. Maps a team name to its flag emoji.
 *
 * How flag emoji work: a country flag is just its ISO 3166-1 alpha-2 code written
 * with "regional indicator" letters (🇩 + 🇪 = 🇩🇪). So instead of pasting 200 raw
 * emoji, we store the 2-letter CODE per team and build the emoji at runtime. Much
 * easier to read, check, and extend.
 *
 * Two things aren't ISO codes:
 *   - England / Scotland / Wales use special subdivision emoji (tag sequences).
 *   - CONIFA sides, micronations, ethnic teams, and defunct states (Sealand,
 *     Yorkshire, Sápmi, Kosovo, Yugoslavia, ...) have NO standard flag emoji, so
 *     they fall back to ⚽. That's honest, not lazy — no emoji exists to show.
 */
window.StatXI = window.StatXI || {};

(function () {
  // Non-ISO subdivision flags (special Unicode tag sequences).
  var SPECIAL = {
    'England':  '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
    'Scotland': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
    'Wales':    '🏴󠁧󠁢󠁷󠁬󠁳󠁿'
  };

  // team name -> ISO 3166-1 alpha-2. Includes dependencies/territories that have
  // their own flag emoji (Aruba, Faroe Islands, Gibraltar, ...). A few CONIFA
  // names are pointed at the matching territory's flag (Ellan Vannin -> Isle of
  // Man, Tahiti -> French Polynesia, Parishes of Jersey -> Jersey).
  var ISO = {
    'Afghanistan':'AF','Albania':'AL','Algeria':'DZ','American Samoa':'AS',
    'Andorra':'AD','Angola':'AO','Anguilla':'AI','Antigua and Barbuda':'AG',
    'Argentina':'AR','Armenia':'AM','Aruba':'AW','Australia':'AU','Austria':'AT',
    'Azerbaijan':'AZ','Bahamas':'BS','Bahrain':'BH','Bangladesh':'BD',
    'Barbados':'BB','Belarus':'BY','Belgium':'BE','Belize':'BZ','Benin':'BJ',
    'Bermuda':'BM','Bhutan':'BT','Bolivia':'BO','Bonaire':'BQ',
    'Bosnia and Herzegovina':'BA','Botswana':'BW','Brazil':'BR',
    'British Virgin Islands':'VG','Brunei':'BN','Bulgaria':'BG','Burkina Faso':'BF',
    'Burundi':'BI','Cambodia':'KH','Cameroon':'CM','Canada':'CA','Cape Verde':'CV',
    'Cayman Islands':'KY','Central African Republic':'CF','Chad':'TD','Chile':'CL',
    'China':'CN','Colombia':'CO','Comoros':'KM','Congo':'CG','Cook Islands':'CK',
    'Costa Rica':'CR','Croatia':'HR','Cuba':'CU','Curaçao':'CW','Cyprus':'CY',
    'Czech Republic':'CZ','DR Congo':'CD','Denmark':'DK','Djibouti':'DJ',
    'Dominica':'DM','Dominican Republic':'DO','Ecuador':'EC','Egypt':'EG',
    'El Salvador':'SV','Ellan Vannin':'IM','Equatorial Guinea':'GQ','Eritrea':'ER',
    'Estonia':'EE','Eswatini':'SZ','Ethiopia':'ET','Falkland Islands':'FK',
    'Faroe Islands':'FO','Fiji':'FJ','Finland':'FI','France':'FR',
    'French Guiana':'GF','Gabon':'GA','Gambia':'GM','Georgia':'GE','Germany':'DE',
    'Ghana':'GH','Gibraltar':'GI','Greece':'GR','Greenland':'GL','Grenada':'GD',
    'Guadeloupe':'GP','Guam':'GU','Guatemala':'GT','Guernsey':'GG','Guinea':'GN',
    'Guinea-Bissau':'GW','Guyana':'GY','Haiti':'HT','Honduras':'HN','Hong Kong':'HK',
    'Hungary':'HU','Iceland':'IS','India':'IN','Indonesia':'ID','Iran':'IR',
    'Iraq':'IQ','Isle of Man':'IM','Israel':'IL','Italy':'IT','Ivory Coast':'CI',
    'Jamaica':'JM','Japan':'JP','Jersey':'JE','Jordan':'JO','Kazakhstan':'KZ',
    'Kenya':'KE','Kiribati':'KI','Kuwait':'KW','Kyrgyzstan':'KG','Laos':'LA',
    'Latvia':'LV','Lebanon':'LB','Lesotho':'LS','Liberia':'LR','Libya':'LY',
    'Liechtenstein':'LI','Lithuania':'LT','Luxembourg':'LU','Macau':'MO',
    'Madagascar':'MG','Malawi':'MW','Malaysia':'MY','Maldives':'MV','Mali':'ML',
    'Malta':'MT','Marshall Islands':'MH','Martinique':'MQ','Mauritania':'MR',
    'Mauritius':'MU','Mayotte':'YT','Mexico':'MX','Micronesia':'FM','Moldova':'MD',
    'Monaco':'MC','Mongolia':'MN','Montenegro':'ME','Montserrat':'MS','Morocco':'MA',
    'Mozambique':'MZ','Myanmar':'MM','Namibia':'NA','Nepal':'NP','Netherlands':'NL',
    'New Caledonia':'NC','New Zealand':'NZ','Nicaragua':'NI','Niger':'NE',
    'Nigeria':'NG','North Korea':'KP','North Macedonia':'MK',
    'Northern Mariana Islands':'MP','Norway':'NO','Oman':'OM','Pakistan':'PK',
    'Palau':'PW','Palestine':'PS','Panama':'PA','Papua New Guinea':'PG',
    'Paraguay':'PY','Parishes of Jersey':'JE','Peru':'PE','Philippines':'PH',
    'Poland':'PL','Portugal':'PT','Puerto Rico':'PR','Qatar':'QA',
    'Republic of Ireland':'IE','Romania':'RO','Russia':'RU','Rwanda':'RW',
    'Réunion':'RE','Saint Barthélemy':'BL','Saint Helena':'SH',
    'Saint Kitts and Nevis':'KN','Saint Lucia':'LC','Saint Martin':'MF',
    'Saint Pierre and Miquelon':'PM','Saint Vincent and the Grenadines':'VC',
    'Samoa':'WS','San Marino':'SM','Saudi Arabia':'SA','Senegal':'SN','Serbia':'RS',
    'Seychelles':'SC','Sierra Leone':'SL','Singapore':'SG','Sint Maarten':'SX',
    'Slovakia':'SK','Slovenia':'SI','Solomon Islands':'SB','Somalia':'SO',
    'South Africa':'ZA','South Korea':'KR','South Sudan':'SS','Spain':'ES',
    'Sri Lanka':'LK','Sudan':'SD','Suriname':'SR','Sweden':'SE','Switzerland':'CH',
    'Syria':'SY','São Tomé and Príncipe':'ST','Tahiti':'PF','Taiwan':'TW',
    'Tajikistan':'TJ','Tanzania':'TZ','Thailand':'TH','Timor-Leste':'TL','Togo':'TG',
    'Tonga':'TO','Trinidad and Tobago':'TT','Tunisia':'TN','Turkey':'TR',
    'Turkmenistan':'TM','Turks and Caicos Islands':'TC','Tuvalu':'TV','Uganda':'UG',
    'Ukraine':'UA','United Arab Emirates':'AE','United States':'US',
    'United States Virgin Islands':'VI','Uruguay':'UY','Uzbekistan':'UZ',
    'Vanuatu':'VU','Vatican City':'VA','Venezuela':'VE','Vietnam':'VN',
    'Wallis Islands and Futuna':'WF','Western Sahara':'EH','Yemen':'YE','Zambia':'ZM',
    'Zimbabwe':'ZW','Åland Islands':'AX'
  };

  // Turn a 2-letter ISO code into its flag emoji by shifting each letter into the
  // regional-indicator block (A -> 🇦 at codepoint 0x1F1E6).
  function iso2emoji(cc){
    return cc.toUpperCase().replace(/./g, function(ch){
      return String.fromCodePoint(0x1F1E6 + ch.charCodeAt(0) - 65);
    });
  }

  StatXI.flag = function(team){
    if (SPECIAL[team]) return SPECIAL[team];
    var cc = ISO[team];
    return cc ? iso2emoji(cc) : '⚽';   // ⚽ = no standard flag emoji exists
  };
})();
