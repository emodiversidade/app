/**
 * RankingVizinhanca.gs — modulo canonico da frota (codex-ranking-vizinhanca)
 *
 * Contrato pedagogico: o estudante nao ve o ranking inteiro, ve a propria
 * vizinhanca — ate 2 colegas com pontuacao MAIOR OU IGUAL a dele e ate 2 com
 * pontuacao MENOR OU IGUAL. A janela de 5 linhas mostra a distancia que da para
 * percorrer na proxima tentativa, em vez de um topo inalcancavel.
 *
 * Funcao pura, sem planilha e sem sessao: recebe a lista ja carregada e
 * devolve a janela. Cada projeto so precisa do adaptador que monta a lista
 * ({ id, nome, pontuacao }) a partir da propria fonte de score.
 *
 * Empate: quem empata com o alvo aparece como vizinho — e o caso mais comum de
 * "maior ou igual". A ordem entre empatados e estavel e alfabetica, para a
 * mesma pontuacao nao trocar de lugar a cada carregamento.
 */

/**
 * @param {!Array<!Object>} entradas [{ id, nome, pontuacao }]
 * @param {*} idAlvo Identificador do jogador no centro da janela.
 * @param {Object=} opcoes { vizinhos: 2, menorMelhor: false }
 *     menorMelhor inverte a ordem para os jogos pontuados por menos
 *     movimentos, menos tempo ou menos erros. "Acima" continua querendo dizer
 *     melhor colocado, que e o que a tela mostra.
 * @return {{posicao: number, total: number, linhas: !Array<!Object>}}
 */
function rankingVizinhanca(entradas, idAlvo, opcoes) {
  var config = opcoes || {};
  var vizinhos = config.vizinhos === undefined ? 2 : Number(config.vizinhos);
  if (!isFinite(vizinhos) || vizinhos < 0) vizinhos = 2;
  var menorMelhor = config.menorMelhor === true;

  var lista = (entradas || []).map(function (entrada, indice) {
    return {
      id: entrada && entrada.id !== undefined ? entrada.id : indice,
      nome: String((entrada && (entrada.nome || entrada.name || entrada.username)) || 'Sem nome'),
      pontuacao: Number((entrada && (entrada.pontuacao !== undefined ? entrada.pontuacao : entrada.score)) || 0)
    };
  }).filter(function (entrada) {
    return isFinite(entrada.pontuacao);
  });

  lista.sort(function (a, b) {
    if (b.pontuacao !== a.pontuacao) {
      return menorMelhor ? a.pontuacao - b.pontuacao : b.pontuacao - a.pontuacao;
    }
    return a.nome.localeCompare(b.nome, 'pt-BR');
  });

  lista.forEach(function (entrada, indice) {
    entrada.posicao = indice + 1;
  });

  var alvo = -1;
  for (var i = 0; i < lista.length; i++) {
    if (String(lista[i].id) === String(idAlvo)) { alvo = i; break; }
  }

  // Sem o alvo na lista (jogador que ainda nao pontuou) a vizinhanca nao existe:
  // devolvemos o topo da mesma janela, que e o que faz sentido mostrar a quem
  // esta comecando.
  if (alvo < 0) {
    return {
      posicao: 0,
      total: lista.length,
      linhas: lista.slice(0, vizinhos * 2 + 1).map(function (entrada) {
        return rankingVizinhancaMarcar_(entrada, 'acima');
      })
    };
  }

  var inicio = Math.max(0, alvo - vizinhos);
  var fim = Math.min(lista.length, alvo + vizinhos + 1);
  var linhas = [];
  for (var j = inicio; j < fim; j++) {
    linhas.push(rankingVizinhancaMarcar_(lista[j], j < alvo ? 'acima' : (j > alvo ? 'abaixo' : 'alvo')));
  }

  return { posicao: lista[alvo].posicao, total: lista.length, linhas: linhas };
}

/** @private */
function rankingVizinhancaMarcar_(entrada, relacao) {
  return {
    posicao: entrada.posicao,
    id: entrada.id,
    nome: entrada.nome,
    pontuacao: entrada.pontuacao,
    relacao: relacao,
    destaque: relacao === 'alvo'
  };
}

/**
 * Testes da funcao pura. Rode no editor do Apps Script; nao toca planilha.
 * @return {!Object}
 */
function testarRankingVizinhanca() {
  var base = [
    { id: 'a', nome: 'Ana', pontuacao: 100 },
    { id: 'b', nome: 'Bruno', pontuacao: 90 },
    { id: 'c', nome: 'Caio', pontuacao: 80 },
    { id: 'd', nome: 'Duda', pontuacao: 80 },
    { id: 'e', nome: 'Elis', pontuacao: 70 },
    { id: 'f', nome: 'Fabio', pontuacao: 60 },
    { id: 'g', nome: 'Gil', pontuacao: 50 }
  ];
  var falhas = [];

  function conferir(rotulo, obtido, esperado) {
    if (String(obtido) !== String(esperado)) {
      falhas.push(rotulo + ': esperado ' + esperado + ', obtido ' + obtido);
    }
  }

  var meio = rankingVizinhanca(base, 'e');
  conferir('meio.total', meio.total, 7);
  conferir('meio.posicao', meio.posicao, 5);
  conferir('meio.linhas', meio.linhas.length, 5);
  conferir('meio.nomes', meio.linhas.map(function (l) { return l.nome; }).join(','),
           'Caio,Duda,Elis,Fabio,Gil');
  conferir('meio.alvo', meio.linhas[2].relacao, 'alvo');

  // Empate: Caio e Duda tem 80; quem empata conta como vizinho.
  var empate = rankingVizinhanca(base, 'c');
  conferir('empate.nomes', empate.linhas.map(function (l) { return l.nome; }).join(','),
           'Ana,Bruno,Caio,Duda,Elis');

  // Borda superior: nao ha 2 acima, a janela encolhe em vez de inventar linha.
  var topo = rankingVizinhanca(base, 'a');
  conferir('topo.linhas', topo.linhas.length, 3);
  conferir('topo.posicao', topo.posicao, 1);

  // Borda inferior.
  var fundo = rankingVizinhanca(base, 'g');
  conferir('fundo.linhas', fundo.linhas.length, 3);

  // Jogador ainda sem pontuacao na lista.
  var forasteiro = rankingVizinhanca(base, 'zzz');
  conferir('forasteiro.posicao', forasteiro.posicao, 0);
  conferir('forasteiro.linhas', forasteiro.linhas.length, 5);

  // menorMelhor: jogo pontuado por menos movimentos inverte a ordem.
  var invertido = rankingVizinhanca(base, 'e', { menorMelhor: true });
  conferir('invertido.posicao', invertido.posicao, 3);
  conferir('invertido.nomes', invertido.linhas.map(function (l) { return l.nome; }).join(','),
           'Gil,Fabio,Elis,Caio,Duda');

  // Lista vazia nao explode.
  var vazio = rankingVizinhanca([], 'a');
  conferir('vazio.total', vazio.total, 0);
  conferir('vazio.linhas', vazio.linhas.length, 0);

  var resultado = { ok: falhas.length === 0, falhas: falhas };
  Logger.log(JSON.stringify(resultado, null, 2));
  return resultado;
}
