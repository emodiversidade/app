/* Catalogo semantico de fixtures analiticos. Gerado a partir do schema real do projeto. */
var AI_FIXTURE_CATALOG = [
  {
    "sheetName": "Situacoes",
    "purpose": "Exercitar reflexoes sobre escolhas e impactos sem dados de estudantes reais.",
    "headers": ["ID","Titulo","Descricao","Opcao1","Opcao2","Resultado1","Resultado2","NivelPrincesamento","ImpactoOpcao1","ImpactoOpcao2"],
    "rows": [
      ["SINT-SIT-001","Trabalho em grupo","O grupo precisa decidir como dividir uma tarefa com pouco tempo.","Propor uma divisão equilibrada e revisar juntos.","Assumir tudo para terminar mais rápido.","A equipe preserva voz, critério e cooperação.","A tarefa termina, mas reduz a participação das outras pessoas.","intermediario","{\"autoconhecimento\":2,\"voz\":3,\"coragem\":1,\"empatia\":2,\"criterio\":3}","{\"autoconhecimento\":1,\"voz\":-1,\"coragem\":2,\"empatia\":-2,\"criterio\":1}"],
      ["SINT-SIT-002","Comentário difícil","Uma colega recebe um comentário que pode ser interpretado de mais de uma forma.","Perguntar como ela se sentiu e reformular com cuidado.","Ignorar o efeito para evitar conflito.","O diálogo cria espaço para revisão e cuidado.","O silêncio reduz o conflito imediato, mas deixa a dúvida aberta.","avancado","{\"autoconhecimento\":3,\"voz\":2,\"coragem\":2,\"empatia\":3,\"criterio\":2}","{\"autoconhecimento\":-1,\"voz\":0,\"coragem\":-1,\"empatia\":-2,\"criterio\":0}"]
    ]
  }
];
